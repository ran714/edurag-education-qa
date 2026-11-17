# 璇ヨ剼鏈敤浜? 鍚戦噺瀛樺偍鍜屾绱?
# 鏍稿績娴佺▼: 鏂囨。鍚戦噺鐢熸垚(BGE-M3) -> 鍚戦噺鍏ュ簱(Milvus) -> 娣峰悎妫€绱?绋犲瘑 + 绋€鐤忓悜閲? -> 缁撴灉鍘婚噸 -> 閲嶆帓搴?-> 杩斿洖绮惧噯鏂囨。.
# 甯哥敤鐨勬绱㈢瓥鐣ヤ粙缁?
#   1. 瀵嗛泦妫€绱? 璇箟绾у尮閰?-> '鑻规灉鍏虫満'鍖归厤'IPhone閲嶅惎'
#   2. 绋€鐤忔绱? 鍏抽敭璇嶇骇杩囨护 -> 鍘诲悎鍚屼腑鎵?杩濈害璐ｄ换'
#   3. 澶氬悜閲忔绱? 閫愯瘝缁嗚妭瀵规瘮 -> 鍖哄垎'浜哄伐鏅鸿兘'鍜?AI'
# 濂藉: 瑕嗙洊'璇箟鐞嗚В-蹇€熻繃婊?缁嗚妭瀵规瘮'鍏ㄥ満鏅? 閫傜敤浜? 鏅鸿兘瀹㈡湇,娉曞緥鏂囨。瀹氫綅,璁烘枃瀵规瘮鍒嗘瀽...

# BGE-M3妯″瀷浠嬬粛: 鏄竴娆捐秴鍏ㄨ兘鐨勬枃鏈?缈昏瘧鍣? + '鎼滅储寮曟搸', 鑳芥妸鏂囧瓧杞垚璇嶅悜閲? 甯绠楁満蹇€熺悊瑙ｅ拰妫€绱俊鎭?


# 妯″瀷璁惧鐩稿叧: 鍒ゆ柇GPU鏄惁鍙敤, 涓烘ā鍨嬮€夋嫨杩愯璁惧.
# 瀵煎叆 hashlib 妯″潡锛岀敤浜庣敓鎴愬敮涓€ ID 鐨勫搱甯屽€?-> 浣滀负Milvus鐨勪富閿?
import hashlib
import json
import os
import sys
from pathlib import Path

import torch.cuda

# 瀵煎叆 Document 绫伙紝鐢ㄤ簬鍒涘缓鏂囨。瀵硅薄, 鍗? 缁熶竴鏂囨。鏁版嵁鏍煎紡(鍚唴瀹?+ 鍏冩暟鎹?
from langchain_core.documents import Document

# 瀵煎叆 BGE-M3 宓屽叆鍑芥暟锛岀敤浜庣敓鎴愭枃妗ｅ拰鏌ヨ鐨勫悜閲忚〃绀?
from milvus_model.hybrid import BGEM3EmbeddingFunction

# 瀵煎叆 Milvus 鐩稿叧绫伙紝鐢ㄤ簬鎿嶄綔鍚戦噺鏁版嵁搴? 渚嬪: 瀹㈡埛绔? 鏁版嵁绫诲瀷, 妫€绱㈣姹? 鎺掑簭鍣?
from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker

# 瀵煎叆 CrossEncoder锛岀敤浜庨噸鎺掑簭鍜?NLI 鍒ゆ柇 -> 浼樺寲妫€绱㈢粨鏋滅殑鐩稿叧鎬ф帓搴?
from sentence_transformers import CrossEncoder

# from document_processor import *        # 瀵煎叆鑷畾涔夌殑鏂囨。鍔犺浇鍜屽垏鍒嗗嚱鏁?


# todo 1. 璺緞閰嶇疆妯″潡. 瀹氫綅褰撳墠鑴氭湰, RAG鏍稿績鐩綍, 椤圭洰鏍圭洰褰? 纭繚鑷畾涔夋ā鍧楀彲瀵煎叆.
# 1.鑾峰彇褰撳墠鏂囦欢鎵€鍦ㄧ洰褰曠殑缁濆璺緞.
local_path = os.path.abspath(os.path.dirname(__file__))
# print(f'local_path: {local_path}')      # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa\core

# 2. 鑾峰彇褰撳墠鐩綍鐨勪笂绾х洰褰?
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
# print(f'module_dir: {rag_qa_path}')     # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa

# 3. 鑾峰彇椤圭洰鏍圭洰褰曠殑缁濆璺緞.
project_root = os.path.dirname(rag_qa_path)
# print(f'project_root: {project_root}')  # D:\workspace\ai_30_bj\integrated_qa_system

# 4. 灏嗛」鐩牴鐩綍鍔犲叆绯荤粺璺緞, 纭繚璺ㄧ洰褰曞鍏ヨ嚜瀹氫箟妯″潡(渚嬪: config, logger)...
sys.path.insert(0, project_root)

from base.config import Config
from base.logger import logger

# todo 2.鍒濆鍖栧叏灞€閰嶇疆 -> 瀹炰緥鍖栭厤缃璞?
conf = Config()


# todo 3.瀹氫箟VectorStore绫? 灏佽鍚戦噺搴撶殑鏍稿績鎿嶄綔 -> 闆嗗悎绠＄悊, 鏂囨。鍏ュ簱, 娣峰悎妫€绱? 缁撴灉澶勭悊.
class VectorStore:
    # todo 3.1 鍒濆鍖栨柟娉?-> 閰嶇疆鍚戦噺搴撹繛鎺? 鍔犺浇妯″瀷, 鍒濆鍖栧鎴风.
    def __init__(
        self,
        collection_name=conf.MILVUS_COLLECTION_NAME,  # Milvus闆嗗悎鍚?-> 绫讳技浜嶴QL鐨勮〃鍚? 杩欓噷鎴戜滑鍙? edurag_final
        host=conf.MILVUS_HOST,  # Milvus涓绘満鍦板潃
        port=conf.MILVUS_PORT,  # Milvus绔彛
        database=conf.MILVUS_DATABASE_NAME,  # Milvus鏁版嵁搴撳悕
    ):
        # 1. 瀛樺偍Milvus鐨勬牳蹇冭缃?
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database

        # 2. 鍒濆鍖栨棩蹇楀疄渚?
        self.logger = logger

        # 3. (鎵╁睍)閫夋嫨妯″瀷杩愯璁惧 -> 浼樺厛浣跨敤GPU, 鏃燝PU鍒欑敤CPU
        torch.set_num_threads(int(os.getenv("EDURAG_CPU_THREADS", "2")))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"浣跨敤璁惧: {self.device}")

        # 4. 鍒濆鍖朆GE-Reranker閲嶆帓搴忔ā鍨?-> 浼樺寲妫€绱㈢粨鏋滅浉鍏虫€ф帓搴?
        # 4.1 鎷兼帴閲嶆帓搴忔ā鍨嬬殑鏈湴璺緞. 鍗? rag_qa/models/bge-reranker-large
        reranker_path = os.path.relpath(
            os.path.join(rag_qa_path, "models", "bge-reranker-large"), os.getcwd()
        )
        # print(f'reranker_path: {reranker_path}')

        # 4.2 鍔犺浇妯″瀷 -> 鎸囧畾杩愯璁惧,  妯″瀷鐢ㄤ簬璁＄畻'鏌ヨ-鏂囨。'鐨勭浉鍏虫€у緱鍒?
        self.reranker = CrossEncoder(reranker_path, device=self.device)

        # 5. 鍒濆鍖朆GE-M3妯″瀷 -> 鐢ㄤ簬鐢熸垚鏂囨。鍜屾煡璇㈢殑鍚戦噺琛ㄧず.
        # 5.1 鎷兼帴妯″瀷鏂囦欢鐨勬湰鍦拌矾寰? 鍗? rag_qa/models/bge-m3
        m3_path = os.path.relpath(os.path.join(rag_qa_path, "models", "bge-m3"), os.getcwd())
        # 5.2 鍔犺浇妯″瀷 -> 妯″瀷鐢ㄤ簬鐢熸垚鏂囨。鍜屾煡璇㈢殑鍚戦噺琛ㄧず.
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,  # 妯″瀷鏈湴璺緞
            use_fp16=(
                self.device == "cuda"
            ),  # GPU鏃跺惎鐢ㄥ崐绮惧害璁＄畻(鍑忓皯鍐呭瓨鍗犵敤, 鎻愬崌閫熷害), CPU鏃剁鐢?
            device=self.device,
        )

        # 6. 鑾峰彇BGE-M3绋犲瘑鍚戦噺鐨勭淮搴? 鍥哄畾杈撳嚭1024缁寸瀵嗗悜閲?
        self.dense_dim = self.embedding_function.dim["dense"]
        # print(f'绋犲瘑鍚戦噺鐨勭淮搴? {self.dense_dim}')  # 缁撴灉: 1024

        # 7. 鍒濆鍖朚ilvus瀹㈡埛绔? 寤虹珛: Milvus鍚戦噺搴撹繛鎺?
        # uri鏍煎紡: http://涓绘満鍦板潃:绔彛鍙? db_name 鎸囧畾瑕佽繛鎺ョ殑鏁版嵁搴?
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)

        # 8. 璋冪敤'绉佹湁'鏂规硶 -> 鍒涘缓鏂伴泦鍚?鑻ヤ笉瀛樺湪) 鎴?鍔犺浇宸叉湁闆嗗悎(鑻ュ瓨鍦?
        self._create_or_load_collection()

    # todo 3.2 绉佹湁鏂规硶: 鍒涘缓鏂伴泦鍚?鑻ヤ笉瀛樺湪) 鎴?鍔犺浇宸叉湁闆嗗悎(鑻ュ瓨鍦?
    def _create_or_load_collection(self):
        # 1. 妫€鏌ユ寚瀹氶泦鍚堟槸鍚﹀凡瀛樺湪(閬垮厤閲嶅鍒涘缓)
        if not self.client.has_collection(self.collection_name):
            # 璧拌繖閲? 璇存槑闆嗗悎涓嶅瓨鍦? 闇€瑕佸垱寤洪泦鍚?
            # 2. 鍒涘缓闆嗗悎 Schema锛岀鐢ㄨ嚜鍔?ID(鎵嬪姩鐢ㄦ枃妗ｅ搱甯屼綔涓轰富閿?锛屽惎鐢ㄥ姩鎬佸瓧娈?
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 娣诲姞 ID 瀛楁锛屼綔涓轰富閿紝VARCHAR 绫诲瀷锛屾渶澶ч暱搴?100
            schema.add_field(
                field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100
            )
            # 娣诲姞鏂囨湰瀛楁锛孷ARCHAR 绫诲瀷锛屾渶澶ч暱搴?65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 娣诲姞绋犲瘑鍚戦噺瀛楁锛孎LOAT_VECTOR 绫诲瀷锛岀淮搴︾敱宓屽叆鍑芥暟鎸囧畾
            schema.add_field(
                field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim
            )
            # 娣诲姞绋€鐤忓悜閲忓瓧娈碉紝SPARSE_FLOAT_VECTOR 绫诲瀷
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 娣诲姞鐖跺潡 ID 瀛楁锛孷ARCHAR 绫诲瀷锛屾渶澶ч暱搴?100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 娣诲姞鐖跺潡鍐呭瀛楁锛孷ARCHAR 绫诲瀷锛屾渶澶ч暱搴?65535
            schema.add_field(
                field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535
            )
            # 娣诲姞瀛︾绫诲埆瀛楁锛孷ARCHAR 绫诲瀷锛屾渶澶ч暱搴?50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 娣诲姞鏃堕棿鎴冲瓧娈碉紝VARCHAR 绫诲瀷锛屾渶澶ч暱搴?50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 3.鍒涘缓绱㈠紩鍙傛暟瀵硅薄
            index_params = self.client.prepare_index_params()
            # 涓虹瀵嗗悜閲忓瓧娈垫坊鍔?IVF_FLAT 绱㈠紩锛屽害閲忕被鍨嬩负鍐呯Н (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )
            # 涓虹█鐤忓悜閲忓瓧娈垫坊鍔?SPARSE_INVERTED_INDEX 绱㈠紩锛屽害閲忕被鍨嬩负鍐呯Н (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={
                    "drop_ratio_build": 0.2
                },  # 鏋勫缓绱㈠紩鏃?涓㈠純20%鐨勪綆鏉冮噸鍊?鍑忓皯瀛樺偍,鍣０, 绮惧害褰卞搷杈冨皬)
            )

            # 4. 鍒涘缓 Milvus 闆嗗悎锛屽簲鐢ㄥ畾涔夌殑 Schema 鍜岀储寮曞弬鏁?
            self.client.create_collection(
                collection_name=self.collection_name, schema=schema, index_params=index_params
            )
            # 5.璁板綍鍒涘缓闆嗗悎鐨勬棩蹇?
            logger.info(f"宸插垱寤洪泦鍚?{self.collection_name}")

        # 6.濡傛灉闆嗗悎宸插瓨鍦?
        else:
            # 璁板綍鍔犺浇闆嗗悎鐨勬棩蹇?
            logger.info(f"宸插姞杞介泦鍚?{self.collection_name}")
        # 7. 灏嗛泦鍚堝姞杞藉埌鍐呭瓨锛岀‘淇濆彲绔嬪嵆鏌ヨ
        self.client.load_collection(self.collection_name)

    # todo 3.3 瀹氫箟鍑芥暟 -> 灏嗘枃妗?瀛愬潡) 杞崲涓?鍚戦噺骞舵彃鍏ilvus
    def add_documents(self, documents):
        # 1. 鎻愬彇鎵€鏈夋枃妗ｅ唴瀹瑰垪琛?
        texts = [
            doc.page_content for doc in documents
        ]  # 鏍煎紡: ['鏂囨。1鍐呭', '鏂囨。2鍐呭', '鏂囨。3鍐呭']

        # 2. 浣跨敤BGE-M3妯″瀷鐢熸垚鏂囨。鍚戦噺.
        # 杈撳叆: 鏂囨湰鍒楄〃(texts)
        # 杈撳嚭: 瀛楀吀, 鍚?dense'(绋犲瘑鍚戦噺) 鍜?'sparse'(绋€鐤忓悜閲?
        embeddings = self.embedding_function(texts)

        # 3. 鍒濆鍖栫┖鍒楄〃.
        data = []

        # 4. 閬嶅巻姣忎釜鏂囨。, 缁勮鎻掑叆鏁版嵁(i -> 鏂囨。绱㈠紩, doc鏄綋鍓嶇殑Document瀵硅薄)
        for i, doc in enumerate(documents):
            # 4.1 鐢熸垚鏂囨。鍞竴鐨処D -> 鏂囨。鍝堝笇鍊?
            # encode('utf-8'):  灏嗗瓧绗︿覆杞垚瀛楄妭娴?-> 鍥犱负MD5鍝堝笇闇€瑕佸瓧鑺?
            # hexdigest():      灏嗗搱甯岀粨鏋滆浆涓?6杩涘埗瀛楃涓?32浣? 閫傚悎浣滀负ID)
            identity = (
                f"{doc.metadata.get('source')}:{doc.metadata.get('file_name')}:{doc.page_content}"
            )
            text_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            # 4.2 澶勭悊绋€鐤忓悜閲? BGE-M3杩斿洖鐨勭█鐤忓悜閲忔槸鐭╅樀, 闇€瑕佽浆鎹负Milvus鏀寔鐨勫瓧鍏告牸寮?
            sparse_vector = {}  # 鍒濆鍖栫█鐤忓悜閲忓瓧鍏?
            # 4.2.1 鑾峰彇绗琲涓枃妗ｇ殑绋€鐤忓悜閲忚 getrow(i) -> 鑾峰彇鐭╅樀鐨勭i琛?
            row = embeddings["sparse"][[i]]
            # (0, 5)	0.029301747679710388  -> (鍒楃储寮? 鏉冮噸)     shape:  (1, 250002)
            # print(f'绋€鐤忓悜閲忚: {row}, shape: {row.shape}')

            # 4.2.2 鑾峰彇绋€鐤忓悜閲忕殑闈為浂鍊肩储寮?鍜?瀵瑰簲鐨勬潈閲?
            indics = row.indices  # 闈為浂鍊肩殑绱㈠紩鍒楄〃, 渚嬪: [10, 25, 50]
            # print(f'闈為浂鍊肩殑绱㈠紩鍒楄〃: {indics}')
            values = row.data  # 闈?   111闆跺€肩殑鏉冮噸鍒楄〃, 渚嬪: [0.8, 0.5, 0.3]

            # 4.2.3 缁勮绋€鐤忓悜閲忓瓧鍏? 灏嗙储寮?鍜?鏉冮噸閰嶅.
            for idx, value in zip(indics, values):  # {10:0.8, 25:0.5, 50:0.3...}
                sparse_vector[idx] = value
                # print(f'绋€鐤忓悜閲忓瓧鍏? {sparse_vector}, 绋€鐤忓悜閲忓瓧鍏搁暱搴? {len(sparse_vector)}')

                # 鎵╁睍: 鏌ョ湅绋犲瘑鍚戦噺鍙婄淮搴?
                # print(f'绋犲瘑鍚戦噺: {embeddings["dense"][i]}, 绋犲瘑鍚戦噺缁村害: {len(embeddings["dense"][i])}')

            # 4.3 缁勮鍗曟潯鎻掑叆鏁版嵁 -> 瀛楁瑕佸拰Schema瀹屽叏瀵瑰簲.
            data.append(
                {
                    "id": text_hash,  # 鍞竴ID(md5鍝堝笇)
                    "text": doc.page_content,  # 鏂囨。鍐呭
                    "dense_vector": embeddings["dense"][i],  # 绋犲瘑鍚戦噺(BGE-M3鐢熸垚)
                    "sparse_vector": sparse_vector,  # 绋€鐤忓悜閲?缁勮鍚庣殑瀛楀吀褰㈠紡)
                    "parent_id": doc.metadata[
                        "parent_id"
                    ],  # 鐖舵枃妗D(浠庡厓鏁版嵁鑾峰彇, 渚嬪: doc_0_parent_0)
                    "parent_content": doc.metadata[
                        "parent_content"
                    ],  # 鐖舵枃妗ｅ唴瀹?浠庡厓鏁版嵁鑾峰彇, 鐢ㄤ簬: 涓婁笅鏂囪ˉ鍏?
                    "source": doc.metadata.get(
                        "source", "unknown"
                    ),  # 瀛︾绫诲埆(浠庡厓鏁版嵁鑾峰彇, 渚嬪: ai)
                    "timestamp": doc.metadata.get("timestamp", "unknown"),
                    "file_name": doc.metadata.get(
                        "file_name", Path(doc.metadata.get("file_path", "资料")).name
                    ),
                }
            )

        # 5. 鎻掑叆鏁版嵁鍒癕ilvus -> 浠呭綋data闈炵┖鏃舵墽琛?
        if data:
            # 浣跨敤upsert鎿嶄綔: 鐩稿悓ID鍒欐洿鏂? 涓嶅瓨鍦ㄥ垯鎻掑叆(閬垮厤閲嶅)
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 璁板綍鏃ュ織
            logger.info(f"宸叉彃鍏?鏇存柊 {len(data)} 鏉℃暟鎹埌闆嗗悎 {self.collection_name}")

    # todo 3.4 瀹氫箟鍑芥暟 -> 娣峰悎妫€绱?绋犲瘑 + 绋€鐤?  + 閲嶆帓搴? 杩斿洖绮惧噯鐖舵枃妗?
    def hybrid_search_with_rerank(
        self,
        query,
        k=conf.RETRIEVAL_K,
        source_filter=None,
        *,
        mode="hybrid",
        rerank=None,
        limit=None,
    ):
        """
        璇ュ嚱鏁扮敤浜庢墽琛? 娣峰悎妫€绱?绋犲瘑 + 绋€鐤忓悜閲? + 缁撴灉閲嶆帓搴? 杩斿洖绮惧噯鐖舵枃妗?
        :param query: 鐢ㄦ埛鏌ヨ鏂囨湰, 渚嬪: 'AI瀛︾鐨勮绋嬪唴瀹规槸浠€涔?
        :param k: 娣峰悎妫€绱㈣繑鍥炵殑TopK瀛愬潡鏁伴噺
        :param source_filter: 瀛︾杩囨护鏉′欢, 渚嬪: ai, 浠呮绱㈣瀛︾鐨勬枃妗? 榛樿None涓嶈繃婊?
        :return: 閲嶆帓搴忓悗鐨凾op-M涓埗鏂囨。鍒楄〃(M鏄粠conf涓幏鍙栫殑)
        """
        # 1.浣跨敤 BGE-M3 宓屽叆鍑芥暟鐢熸垚鏌ヨ鐨勫祵鍏?
        query_embeddings = self.embedding_function([query])
        # 2.鑾峰彇鏌ヨ鐨勭瀵嗗悜閲?
        dense_query_vector = query_embeddings["dense"][0]

        # 3.鍒濆鍖栨煡璇㈢殑绋€鐤忓悜閲忓瓧鍏?
        sparse_query_vector = {}
        # 3.1 鑾峰彇鏌ヨ绋€鐤忓悜閲忕殑绗?0 琛屾暟鎹?-> 浠?涓煡璇? 鏁呭彇绗?琛?
        row = query_embeddings["sparse"][[0]]
        # 3.2 鑾峰彇绋€鐤忓悜閲忕殑闈為浂鍊肩储寮?
        indices = row.indices  # 绱㈠紩鍒楄〃
        # 3.3 鑾峰彇绋€鐤忓悜閲忕殑闈為浂鍊?
        values = row.data  # 鏉冮噸鍒楄〃
        # 3.4 灏嗙储寮曞拰鍊奸厤瀵癸紝濉厖绋€鐤忓悜閲忓瓧鍏?
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        # 4. 鏋勫缓妫€绱㈣繃婊よ〃杈惧紡: 鎸夊绉戣繃婊?-> 鑻ource_filter瀛樺湪.
        filter_expr = (
            f"source == {json.dumps(source_filter, ensure_ascii=False)}" if source_filter else ""
        )
        # print(f'filter_expr: {filter_expr}')

        # 5. 鏋勫缓绋犲瘑鍚戦噺妫€绱㈣姹? 瀹氫箟绋犲瘑鍚戦噺鐨勬绱㈠弬鏁?
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],  # 鏌ヨ鍚戦噺, 鍒楄〃鏍煎紡.
            anns_field="dense_vector",  # 妫€绱㈢殑鍚戦噺瀛楁
            param={
                "metric_type": "IP",
                "params": {"nprobe": 10},
            },  # 鍐呯Н鐩镐技搴? 妫€绱㈡椂璁块棶鐨勮仛绫绘暟(骞宠　閫熷害鍜岀簿搴?
            limit=k,  # 杩斿洖Tok-K缁撴灉
            expr=filter_expr,  # 搴旂敤杩囨护琛ㄨ揪寮?
        )

        # 6. 鏋勫缓绋€鐤忓悜閲忔绱㈣姹? 鏋勫缓绋€鐤忓悜閲忕殑妫€绱㈠弬鏁?
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],  # 鍒楄〃鏍煎紡.
            anns_field="sparse_vector",  # 妫€绱㈢殑鍚戦噺瀛楁
            param={
                "metric_type": "IP",
                "params": {},
            },  # 绋€鐤忓悜閲忔棤闇€棰濆鍙傛暟, 鍙渶瑕佹寚瀹氬唴绉浉浼煎害
            limit=k,  # 妫€绱ok-K缁撴灉
            expr=filter_expr,  # 搴旂敤杩囨护琛ㄨ揪寮?
        )

        # 7. 鍒涘缓鍔犳潈鎺掑簭鍣?-> 铻嶅悎绋犲瘑鍜岀█鐤忔绱㈢殑缁撴灉, 鎸夋潈閲嶈绠楁渶缁堝緱鍒?
        # 鏉冮噸閫昏緫: 绋犲瘑鍚戦噺渚ч噸璇箟鐩镐技搴? 绋€鐤忓悜閲忎晶閲嶅叧閿瘝鍖归厤, 鍙牴鎹笟鍔¤皟鏁?
        ranker = WeightedRanker(1.0, 0.7)  # 绋犲瘑鍚戦噺鏉冮噸: 1.0, 绋€鐤忓悜閲忔潈閲? 0.7
        if mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError("mode must be dense, sparse or hybrid")
        requests = (
            [dense_request, sparse_request]
            if mode == "hybrid"
            else [dense_request if mode == "dense" else sparse_request]
        )
        if mode != "hybrid":
            ranker = WeightedRanker(1.0)
        final_limit = limit if limit is not None else conf.CANDIDATE_M

        # 8. 鎵ц娣峰悎妫€绱?-> 鑾峰彇绋犲瘑鍜岀█鐤忓悜閲忕殑妫€绱㈢粨鏋?, 杩斿洖Top-K瀛愬潡
        results = self.client.hybrid_search(
            collection_name=self.collection_name,  # 鐩爣闆嗗悎
            reqs=requests,
            ranker=ranker,  # 鍔犳潈鎺掑簭鍣?
            limit=k,  # 妫€绱ok-K缁撴灉
            output_fields=[
                "text",
                "parent_id",
                "parent_content",
                "source",
                "timestamp",
                "file_name",
            ],
        )[0]  # 鍥犱负灏?涓煡璇? 鎵€浠ヨ繑鍥炵0涓厓绱?
        # print(f'results: {results}, 绫诲瀷: {type(results)}, 鏁版嵁閲? {len(results)}')    # 鍒楄〃绫诲瀷.

        # 9. 灏嗘绱㈢粨鏋滆浆鎹负 LangChain Document瀵硅薄 -> 缁熶竴鏍煎紡, 渚夸簬鍚庣画澶勭悊.
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        # print(f'瀛愬潡鍒楄〃: {sub_chunks}, 鏁版嵁閲? {len(sub_chunks)}')      # [Document瀵硅薄, Document瀵硅薄, ...]

        # 10. 浠庡瓙鍧椾腑鎻愬彇鍘婚噸鐨勭埗鏂囨。 -> 閬垮厤鍚屼竴鐖跺潡鐨勫涓瓙鍧楅噸澶嶈繑鍥?
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # print(f'(鍘婚噸鍚?鐖舵枃妗ｅ垪琛? {parent_docs}, 鏁版嵁閲? {len(parent_docs)}')  # 鏁版嵁閲? 1

        # 11. 閲嶆帓搴忛€昏緫: 鐖舵枃妗ｆ暟閲?2鏃惰烦杩囨帓搴?鏃犻渶浼樺寲) -> 鐩存帴杩斿洖.
        rerank = conf.RERANK_ENABLED if rerank is None else rerank
        if not rerank or len(parent_docs) < 2:
            return parent_docs[:final_limit]

        # 12. 鐖舵枃妗ｆ暟閲?>= 2鏃? 鎵ц閲嶆帓搴? 鎻愬崌鐩稿叧鎬?
        if parent_docs:
            # 12.1 鏋勫缓'鏌ヨ-鏂囨。'閰嶅鍒楄〃, 閲嶆帓搴忔ā鍨嬮渶瑕佹敼鏍煎紡杈撳叆, 姣忎釜閰嶅涓?[query, doc_content]
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 12.2 璁＄畻鐩稿叧寰楀垎.
            scores = self.reranker.predict(pairs)
            # print(f'鐩稿叧寰楀垎: {scores}')
            # 12.3 鎸夊緱鍒嗛檷搴忔帓搴?-> 灏嗗緱鍒嗕笌鐖舵枃妗ｉ厤瀵? 鎸夊緱鍒嗕粠楂樺埌浣庢帓搴?
            ranked_parent_docs = [
                doc
                for _, doc in sorted(
                    zip(scores, parent_docs), key=lambda item: float(item[0]), reverse=True
                )
            ]
        else:
            # 12.4 鑻ョ埗鏂囨。鍒楄〃涓虹┖, 杩斿洖绌哄垪琛?
            ranked_parent_docs = []

        # 13. 杩斿洖閲嶆帓搴忓悗鐨勭埗鏂囨。鍒楄〃 -> Top-M涓埗鏂囨。
        return ranked_parent_docs[:final_limit]

    # todo 3.5 瀹氫箟鍑芥暟 -> 浠庡瓙鍧楀垪琛ㄤ腑鎻愬彇鍘婚噸鐨勭埗鏂囨。.
    def _get_unique_parent_docs(self, sub_chunks):
        # 鍒濆鍖栭泦鍚堬紝鐢ㄤ簬瀛樺偍宸插鐞嗙殑鐖跺潡鍐呭锛堝幓閲嶏級
        parent_contents = set()
        # 鍒濆鍖栧垪琛紝鐢ㄤ簬瀛樺偍鍞竴鐖舵枃妗?
        unique_docs = []
        # 閬嶅巻鎵€鏈夊瓙鍧?
        for chunk in sub_chunks:
            # 鑾峰彇瀛愬潡鐨勭埗鍧楀唴瀹癸紝榛樿涓哄瓙鍧楀唴瀹?
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 妫€鏌ョ埗鍧楀唴瀹规槸鍚﹂潪绌轰笖鏈噸澶?
            if parent_content and parent_content not in parent_contents:
                # 鍒涘缓鏂扮殑 Document 瀵硅薄锛屽寘鍚埗鍧楀唴瀹瑰拰鍏冩暟鎹?
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 灏嗙埗鍧楀唴瀹规坊鍔犲埌鍘婚噸闆嗗悎
                parent_contents.add(parent_content)
        # 杩斿洖鍘婚噸鍚庣殑鐖舵枃妗ｅ垪琛?
        return unique_docs

    # todo 3.6 瀹氫箟鍑芥暟 -> 灏哅ilvus缁撴灉(hit) 杞崲涓?LangChain Document瀵硅薄.
    def _doc_from_hit(self, hit):
        # 鍒涘缓骞惰繑鍥?Document 瀵硅薄锛屽～鍏呭唴瀹瑰拰鍏冩暟鎹?
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp"),
                "file_name": hit.get("file_name"),
            },
        )


# todo 4.娴嬭瘯浠ｇ爜.
if __name__ == "__main__":
    # 1. 瀹炰緥鍖朧ectorStore瀵硅薄 -> 鑷姩杩炴帴Milvus, 鍒涘缓/鍔犺浇闆嗗悎.
    vector_store = VectorStore()

    # 2.鍚戦噺鍏ュ簱鍔ㄤ綔
    # # 2.1 瀹氫箟閬嶅巻, 璁板綍: 鏂囨。鐩綍.
    # directory_path = '../data/ai_data'
    # # 瀵嗛泦鍚戦噺dense -> 鎶撴暣浣撹涔? 閬垮厤'鍏抽敭璇嶄笉鍚屼絾鏄剰鎬濈浉杩?鐨勬枃鏈婕忔帀.
    # # COLBERT -> 鎶撹瘝绾т笂涓嬫枃, 瑙ｅ喅'鍚岃瘝寮備箟'鐨勫尮閰嶉棶棰?
    # # 绋€鐤忓悜閲弒parse -> 鎶撳叧閿瘝鏉冮噸, 纭繚'鐗瑰畾鏈'鑳界簿鍑嗗懡涓?
    # # embedding_function.dim -> {'dense': 1024, 'colbert_vecs': 1024, 'sparse': 250002}
    # # print(f'embedding_function.dim -> {vector_store.embedding_function.dim}')
    #
    # # 2.2 鑾峰彇鎵€鏈夌殑澶勭悊鍚庣殑鏂囨。.
    # documents = process_documents(directory_path)       # 79涓潡
    # # 2.3 娣诲姞鏂囨。鍒癕ilvus涓?
    # vector_store.add_documents(documents)

    # 3.娣峰悎妫€绱?+ 缁撴灉閲嶆媿, 渚嬪: 鏌ヨ'AI瀛︾鐨勮绋嬪唴瀹规槸浠€涔?, 杩囨护AI瀛︾鏂囨。.
    # query = 'LLM是什么?'
    query = "What is AI?"
    # query = '浣犺璇嗗く鍝ュ悧?'
    results = vector_store.hybrid_search_with_rerank(query, source_filter="ai")

    # 鎵撳嵃妫€绱㈠皢璇惰繃, 鏌ョ湅缁撴灉鏁伴噺 鍜?鍏蜂綋鍐呭.
    print(f"results: {results}")  # 缁撴灉鍒楄〃: Document瀵硅薄
    print(f"results鏁伴噺: {len(results)}")  # 缁撴灉鏁伴噺: 0 ~ 2鍖洪棿
