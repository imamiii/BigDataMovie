# network_builder.py
import networkx as nx
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from config import RATING_THRESHOLD, MIN_COOPERATION_COUNT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkBuilder:
    def __init__(self):
        self.G = nx.Graph()
        self.G_low_rating = None
        self.persons_dict = None

    def build_network(self, edges_df: pd.DataFrame, persons_dict=None):
        """从边数据构建网络：边属性包含 movies(全量) / top_movies / weight / avg_rating"""
        logger.info("开始构建影人合作网络...")
        logger.info(f"边数据大小: {len(edges_df)} 条记录")

        if persons_dict is not None:
            self.persons_dict = persons_dict

        if edges_df is None or edges_df.empty:
            self.G.clear()
            return self.G

        df = edges_df.copy()

        # 清理：去掉自己和自己
        df = df[df["person1"] != df["person2"]].copy()

        # rating 转数值
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        # movie_id 兜底
        if "movie_id" not in df.columns:
            df["movie_id"] = df.index

        # year 兜底
        if "year" not in df.columns:
            df["year"] = 0

        logger.info("正在聚合边数据（按 person1, person2）...")

        grouped = df.groupby(["person1", "person2"], sort=False)

        # 基本统计
        agg_df = grouped.agg(
            weight=("movie_id", "count"),
            avg_rating=("rating", "mean"),
        ).reset_index()

        # movies 全量列表（去重：同一对人同一部电影只保留一次）
        def build_movies_list(g: pd.DataFrame):
            # 先按 movie_id 去重（不包含 types，避免 list 无法 hash）
            cols_basic = ["movie_id", "movie_name", "rating", "year"]
            g2 = g[cols_basic + (["types"] if "types" in g.columns else [])].copy()

            # movie_id 兜底
            if "movie_id" not in g2.columns:
                g2["movie_id"] = g2.index

            # drop_duplicates 只用可 hash 的列
            g2 = g2.drop_duplicates(subset=["movie_id"])

            # types 兜底：保证是 list[str]
            def norm_types(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return []
                if isinstance(x, list):
                    return x
                s = str(x).strip()
                if not s or s in ("未知", "nan"):
                    return []
                for sep in ["、", ",", "，", "|"]:
                    s = s.replace(sep, "/")
                return [p.strip() for p in s.split("/") if p.strip()]

            if "types" in g2.columns:
                g2["types"] = g2["types"].apply(norm_types)
            else:
                g2["types"] = [[] for _ in range(len(g2))]

            # 输出字段统一
            movies = g2[["movie_id", "movie_name", "rating", "year", "types"]].to_dict("records")
            return movies

        movies_series = grouped.apply(build_movies_list)
        # align
        agg_df["movies"] = movies_series.values

        # top_movies：从 movies 里按评分降序取前 5（NaN 排最后）
        def top5(movies):
            def keyfn(x):
                r = x.get("rating", None)
                if r is None or (isinstance(r, float) and np.isnan(r)):
                    return -1e9
                return float(r)
            return sorted(movies, key=keyfn, reverse=True)[:5]

        agg_df["top_movies"] = agg_df["movies"].apply(top5)

        # 节点集合
        all_persons = set(df["person1"]).union(set(df["person2"]))

        # 构建图
        self.G.clear()
        self.G.add_nodes_from(all_persons)

        edges_to_add = []
        for _, row in agg_df.iterrows():
            edges_to_add.append((
                row["person1"], row["person2"],
                {
                    "weight": int(row["weight"]),
                    "avg_rating": row["avg_rating"],
                    "movies": row["movies"],         # ✅ 全量
                    "top_movies": row["top_movies"], # ✅ 仍保留
                }
            ))

        self.G.add_edges_from(edges_to_add)

        logger.info(f"网络构建完成: {self.G.number_of_nodes()} 个节点, {self.G.number_of_edges()} 条边")
        return self.G

    def extract_low_rating_subnetwork(self, threshold=RATING_THRESHOLD):
        """提取低分电影子网络（按边 avg_rating 阈值，仅作为一个预先子网功能保留）"""
        logger.info(f"提取低分电影子网络 (阈值={threshold})...")
        low_rating_edges = []
        for u, v, data in self.G.edges(data=True):
            avg_rating = data.get("avg_rating", np.nan)
            if not pd.isna(avg_rating) and avg_rating <= threshold:
                low_rating_edges.append((u, v))

        if low_rating_edges:
            self.G_low_rating = self.G.edge_subgraph(low_rating_edges).copy()
            logger.info(
                f"低分电影子网络: {self.G_low_rating.number_of_nodes()} 个节点, "
                f"{self.G_low_rating.number_of_edges()} 条边"
            )
        else:
            self.G_low_rating = nx.Graph()
            logger.warning("未找到低分电影边")

        return self.G_low_rating

    def get_high_cooperation_pairs(self, min_count=MIN_COOPERATION_COUNT):
        high_coop_pairs = []
        for u, v, data in self.G.edges(data=True):
            if data.get("weight", 0) >= min_count:
                pair_info = {
                    "person1": u,
                    "person2": v,
                    "cooperation_count": int(data.get("weight", 0)),
                    "avg_rating": data.get("avg_rating", np.nan),
                    "movies": data.get("top_movies", []),
                    "movies_count": len(data.get("top_movies", [])),
                }
                high_coop_pairs.append(pair_info)

        high_coop_pairs.sort(key=lambda x: x["cooperation_count"], reverse=True)
        logger.info(f"找到 {len(high_coop_pairs)} 对合作次数 ≥ {min_count} 的影人组合")
        return high_coop_pairs

    def calculate_network_metrics(self):
        metrics = {}
        metrics["total_nodes"] = self.G.number_of_nodes()
        metrics["total_edges"] = self.G.number_of_edges()
        metrics["density"] = nx.density(self.G) if metrics["total_nodes"] > 1 else 0
        metrics["avg_clustering"] = nx.average_clustering(self.G) if metrics["total_nodes"] > 0 else 0

        connected_components = list(nx.connected_components(self.G))
        metrics["connected_components"] = len(connected_components)
        metrics["largest_component_size"] = max((len(c) for c in connected_components), default=0)

        degrees = [d for _, d in self.G.degree()]
        metrics["avg_degree"] = float(np.mean(degrees)) if degrees else 0.0
        metrics["max_degree"] = int(max(degrees)) if degrees else 0

        degree_centrality = nx.degree_centrality(self.G) if metrics["total_nodes"] > 1 else {}
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        metrics["top_central_nodes"] = top_nodes

        if self.G_low_rating and self.G_low_rating.number_of_nodes() > 0:
            metrics["low_rating_nodes"] = self.G_low_rating.number_of_nodes()
            metrics["low_rating_edges"] = self.G_low_rating.number_of_edges()
            metrics["low_rating_density"] = nx.density(self.G_low_rating) if self.G_low_rating.number_of_nodes() > 1 else 0
            metrics["low_rating_avg_clustering"] = nx.average_clustering(self.G_low_rating) if self.G_low_rating.number_of_nodes() > 0 else 0

        return metrics

    def detect_communities(self, method="louvain"):
        try:
            if method == "louvain":
                import community as community_louvain
                partition = community_louvain.best_partition(self.G)
                logger.info(f"Louvain算法检测到 {len(set(partition.values()))} 个社区")
                return partition

            elif method == "girvan_newman":
                comp = nx.community.girvan_newman(self.G)
                communities = tuple(sorted(c) for c in next(comp))
                logger.info(f"Girvan-Newman算法检测到 {len(communities)} 个社区")
                return communities

            logger.warning(f"不支持的社区检测方法: {method}")
            return None

        except ImportError as e:
            logger.error(f"社区检测失败，请安装相应库: {e}")
            return None

    def get_node_role(self, node):
        """节点角色：如果当过导演则 director，否则如果当过演员则 actor，否则 unknown"""
        if self.persons_dict is None or node not in self.persons_dict:
            return "unknown"

        for movie in self.persons_dict[node]:
            if movie.get("role") == "director":
                return "director"
        for movie in self.persons_dict[node]:
            if movie.get("role") == "actor":
                return "actor"
        return "unknown"
