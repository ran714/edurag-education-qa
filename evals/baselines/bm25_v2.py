# 该脚本用于: BM25搜索, 阈值: >= 0.85才会走MySQL系统.

# 导包
import re
from difflib import SequenceMatcher  # 导入数值计算库

import numpy as np
from rank_bm25 import BM25Okapi  # 导入 BM25 算法

from base import logger  # 导入日志
from mysql_qa.utils.preprocess import preprocess_text  # 导入文本预处理


# todo 1. 定义BM25搜索类 -> 封装数据加载, 模型初始化, 相似度计算, 答案检索等...
class BM25Search:
    # todo 1.1 初始化函数 -> 客户端实例, 模型参数, 加载数据.
    def __init__(self, redis_client, mysql_client):
        # 1. 日志实例.
        self.logger = logger
        # 2. Redis客户端实例.
        self.redis_client = redis_client
        # 3. MySQL客户端实例.
        self.mysql_client = mysql_client
        # 4. 初始化BM25模型.
        self.bm25 = None
        # 5. 初始化分词后的问题列表.
        self.questions = None
        # 6. 初始化原始问题列表.
        self.original_questions = None
        # 7. 调用数据加载方法 -> 加载问题数据, 并初始化模型.
        self._load_data()  # 小细节: 加_的函数, 表示: 内置函数

    # todo 1.2 数据加载方法 -> 优先从Reids缓存加载, 缓存未命中则从MySQL加载.
    def _load_data(self):
        # 1. 定义Redis缓存键 -> 区分原始问题 和 分词后问题的缓存.
        original_key = "qa_original_questions"  # 原始问题缓存键, 它的数据格式为: ['问题1', '问题2', '问题3'...]
        tokenized_key = "qa_tokenized_questions"  # 分词后问题缓存键, 它的数据格式为: [['问题1切词后的词1', '词2', '词3'...], ['问题2切词词1', '词2'...], ['问题3...']...]

        # 2. 从Redis中获取原始问题 -> 缓存读取速度快, 减少数据库访问.
        self.original_questions = self.redis_client.get_data(original_key)
        # print(f'original_questions: {self.original_questions}')

        # 3. 从Redis中获取分词后的问题.
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        # print(f'tokenized_questions: {tokenized_questions}')

        # 4. 若Redis缓存中无数据(或者数据不完整) -> 则从MySQL加载并更新缓存.
        if not self.original_questions or not tokenized_questions:
            # 4.1 从MySQL获取原始文件列表, 格式为 元组嵌套, 例如: ((问题1, ), (问题2, )...)
            self.original_questions = self.mysql_client.fetch_questions()
            # print(f'original_questions: {self.original_questions}')

            # 4.2 若MySQL也无问题数据, 记录警告日志并返回.
            if not self.original_questions:
                self.logger.warning("未从MySQL加载到任何问题数据")
                return

            # 4.3 对原始问题进行分词处理 -> 转换为词语列表.
            # 注意: q[0]是因为原始问题格式为: (('问题1', ), ('问题2', )...)
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]

            # 4.4 将原始问题列表转换为字符串 -> 去除元组嵌套, 并存换到Redis中.
            self.redis_client.set_data(original_key, [(q[0]) for q in self.original_questions])
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        # 5. 保存分词后的问题列表 -> 用于BM25模型初始化.
        self.questions = tokenized_questions
        # 6. 初始化BM25模型.
        self.bm25 = BM25Okapi(self.questions)
        # 7. 记录INFO日志
        self.logger.info("BM25模型 初始化成功")

    # todo 1.3 softmax分数归一化方法 -> 将BM25分数转换为概率分布, 便于阈值判断.
    def _softmax(self, scores):
        # 对输入分数进行softmax归一化, 输出总和为1的概率分布.
        # 1. 计算指数分数: 每个分数减去最大值 -> 防止数值过大导致溢出.
        exp_scores = np.exp(scores - np.max(scores))
        # 2. 返回归一化结果.
        return exp_scores / np.sum(exp_scores)

    # todo 1.4 核心搜索方法: 处理查询, 计算相似度, 返回匹配答案 或 提示无结果.
    @staticmethod
    def _normalize_source_filter(source_filter):
        if not source_filter:
            return None
        value = str(source_filter).strip()
        aliases = {
            "python": "Python学科",
            "python学科": "Python学科",
            "java": "JAVA学科",
            "java学科": "JAVA学科",
        }
        return aliases.get(value.lower(), value)

    def search(self, query, threshold=0.85, source_filter=None):
        """
        根据输入查询检索 最相似的问题 并返回对应答案
        :param query: 用户查询文本
        :param threshold: 相似度阈值 -> 超过此值即为匹配成功
        :return: 匹配成功: (答案, False),  未匹配: (None, True)      True:新查询, False: 非新查询(命中缓存)
        """
        # 1. 检查查询有效性 -> 空或者非字符串类型认为无效.
        if not query or not isinstance(query, str):
            self.logger.error("无效查询: 查询为空或者非字符串类型")
            return None, True

        source_filter = self._normalize_source_filter(source_filter)

        # 缓存键必须包含学科，否则同一问题会复用其他学科的答案。
        cache_scope = source_filter or "all"
        cache_key = f"answer:v2:{cache_scope}:{query}"
        cached_answer = self.redis_client.get_data(cache_key)
        if cached_answer:
            # 走这里, 说明缓存命中(即: 从Redis中获取的答案), 返回即可.
            return cached_answer, False  # 这里的False意思是: 非新查询(命中缓存)

        # 走到这里, 说明: 缓存未命中, 需要从数据库中查询.
        try:
            # 指定学科时构建该学科的临时 BM25 语料；未指定时复用全量索引。
            original_questions = self.original_questions
            bm25 = self.bm25
            if source_filter:
                original_questions = self.mysql_client.fetch_questions(source_filter)
                if not original_questions:
                    self.logger.info(f"学科 {source_filter} 暂无标准问答数据")
                    return None, True
                filtered_tokens = [preprocess_text(q[0]) for q in original_questions]
                bm25 = BM25Okapi(filtered_tokens)

            # 3.对查询文本进行预处理 -> 分词, 和问题库预处理逻辑一致.
            query_tokens = preprocess_text(query)
            # 4. 计算查询 与 (数据库中)所有问题的BM25相似度.
            scores = bm25.get_scores(query_tokens)
            # print(f'scores: {scores}')
            # 5. 对分数进行归一化处理 -> 转换为概率分布.
            # Do not compare a softmax probability with the match threshold.
            # Softmax only expresses relative scores and causes unrelated questions to look highly confident.
            best_idx = int(np.argmax(scores))
            raw_score = float(scores[best_idx])
            best_question = original_questions[best_idx]
            if isinstance(best_question, (tuple, list)):
                best_question = best_question[0]

            def normalize_text(value):
                value = re.sub(r"\s+", "", str(value).lower())
                return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", value)

            normalized_query = normalize_text(query)
            normalized_question = normalize_text(best_question)
            query_tokens = {t for t in query_tokens if re.search(r"[0-9a-z\u4e00-\u9fff]", t)}
            question_tokens = {
                t for t in preprocess_text(best_question) if re.search(r"[0-9a-z\u4e00-\u9fff]", t)
            }
            overlap = len(query_tokens & question_tokens)
            token_recall = overlap / len(query_tokens) if query_tokens else 0.0
            token_precision = overlap / len(question_tokens) if question_tokens else 0.0
            token_f1 = (
                2 * token_recall * token_precision / (token_recall + token_precision)
                if token_recall + token_precision
                else 0.0
            )
            text_similarity = SequenceMatcher(None, normalized_query, normalized_question).ratio()
            exact_match = bool(normalized_query and normalized_query == normalized_question)
            reliable_match = (
                exact_match
                or text_similarity >= 0.82
                or (token_f1 >= 0.72 and token_recall >= 0.72)
            )
            best_score = text_similarity if reliable_match else 0.0
            print(
                f"best_idx: {best_idx}, raw_score: {raw_score:.3f}, "
                f"text_similarity: {text_similarity:.3f}, token_f1: {token_f1:.3f}, "
                f"reliable: {reliable_match}"
            )

            # Only reliable lexical matches may return a MySQL answer.
            if reliable_match and best_score >= min(threshold, 0.85):
                answer = self.mysql_client.fetch_answer(best_question, source_filter)
                if answer:
                    self.redis_client.set_data(cache_key, answer)
                    self.logger.info(f"????, ???: {best_score:.3f}")
                    return answer, False

            # 8. 若未超过阈值, 记录INFO级日志并返回结果.
            self.logger.info(
                f"未找到可靠答案, 最高匹配度: {best_score:.3f}(低于阈值{threshold:.3f})"
            )
            return None, True  # True: 新查询,需要进一步处理(即: 走RAG)

        except Exception as e:
            self.logger.error(f"数据库查询异常: {e}")
            return None, True


# todo 2. 测试代码.
if __name__ == "__main__":
    from mysql_qa.cache.redis_client import RedisClient
    from mysql_qa.db.mysql_client import MySQLClient

    # 1. 实例化Redis 和 MySQL客户端.
    redis_client = RedisClient()
    mysql_client = MySQLClient()

    # 2. 创建BM25搜索实例.
    bm25_search = BM25Search(redis_client, mysql_client)

    # 3. 测试搜索功能, 例如: 查询 'VMware安装VMware时显示灰色如何解决' 或者 PyCharm使用中文版本还是英文版本 或者 Python好学吗
    result = bm25_search.search(query="Python好学吗")
    print(f"result: \n{result}")
