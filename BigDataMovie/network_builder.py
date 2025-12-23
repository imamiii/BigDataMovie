# network_builder.py
import networkx as nx
import pandas as pd
import numpy as np
import logging
from collections import Counter
from config import RATING_THRESHOLD, MIN_COOPERATION_COUNT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkBuilder:
    def __init__(self):
        self.G = nx.Graph()
        self.G_low_rating = None
        self.persons_dict = None

    def build_network(self, edges_df: pd.DataFrame, persons_dict=None):
        """从边数据构建网络：边属性包含 movies(全量) / top_movies / weight / avg_rating / role_pairs"""
        logger.info("开始构建影人合作网络...")
        logger.info(f"边数据大小: {len(edges_df) if edges_df is not None else 0} 条记录")

        if persons_dict is not None:
            self.persons_dict = persons_dict

        if edges_df is None or edges_df.empty:
            self.G.clear()
            return self.G

        df = edges_df.copy()

        # 兜底列
        if "movie_id" not in df.columns:
            df["movie_id"] = df.index
        if "movie_name" not in df.columns:
            df["movie_name"] = df.get("name", df["movie_id"].astype(str))
        if "year" not in df.columns:
            df["year"] = 0
        if "role1" not in df.columns:
            df["role1"] = "unknown"
        if "role2" not in df.columns:
            df["role2"] = "unknown"

        # 清理：去掉自己和自己（同名同人多角色也不建自环）
        df = df[df["person1"] != df["person2"]].copy()

        # rating 转数值
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        # -----------------------------
        # 关键改动：无向边“规范化”
        # 把 (person1, person2) 与 (person2, person1) 统一为同一对 (person_a, person_b)
        # 同时把 role 也跟着对齐到 a/b
        # -----------------------------
        p1 = df["person1"].astype(str)
        p2 = df["person2"].astype(str)
        mask = p1 <= p2  # 字典序，足够稳定；更严格可换成自定义 compare

        df["person_a"] = np.where(mask, p1, p2)
        df["person_b"] = np.where(mask, p2, p1)
        df["role_a"] = np.where(mask, df["role1"].astype(str), df["role2"].astype(str))
        df["role_b"] = np.where(mask, df["role2"].astype(str), df["role1"].astype(str))

        logger.info("正在聚合边数据（按 person_a, person_b）...")

        grouped = df.groupby(["person_a", "person_b"], sort=False)

        # 基本统计
        agg_df = grouped.agg(
            weight=("movie_id", "count"),
            avg_rating=("rating", "mean"),
        ).reset_index()

        # movies 全量列表（去重：同一对人同一部电影只保留一次）
        def build_movies_list(g: pd.DataFrame):
            cols_basic = ["movie_id", "movie_name", "rating", "year", "role_a", "role_b"]
            if "types" in g.columns:
                cols_basic.append("types")

            g2 = g[cols_basic].copy()

            # 同一 movie 只留一条（避免重复边导致 movies 里重复）
            g2 = g2.drop_duplicates(subset=["movie_id"])

            # types 兜底：保证是 list[str]
            def norm_types(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return []
                if isinstance(x, list):
                    # 已经是 list[str] 就直接用
                    return [str(p).strip() for p in x if str(p).strip()]
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

            # 注意：person_a/person_b 在一个 group 内是常量
            pa = g["person_a"].iloc[0]
            pb = g["person_b"].iloc[0]

            records = []
            for _, r in g2.iterrows():
                rec = {
                    "movie_id": r["movie_id"],
                    "movie_name": r["movie_name"],
                    "rating": r["rating"],
                    "year": r["year"],
                    "types": r.get("types", []),
                    # 关键：保留该电影里两人的“当次身份”
                    "roles": {
                        pa: r.get("role_a", "unknown"),
                        pb: r.get("role_b", "unknown"),
                    }
                }
                records.append(rec)

            return records

        movies_series = grouped.apply(build_movies_list)
        agg_df["movies"] = movies_series.values

        # role_pairs：统计这对人出现过哪些角色组合（director-actor / actor-director / director-director 等）
        def build_role_pairs(g: pd.DataFrame):
            s = (g["role_a"].astype(str) + "-" + g["role_b"].astype(str))
            return s.value_counts().to_dict()

        role_pairs_series = grouped.apply(build_role_pairs)
        agg_df["role_pairs"] = role_pairs_series.values

        # top_movies：从 movies 里按评分降序取前 5（NaN 排最后）
        def top5(movies):
            def keyfn(x):
                r = x.get("rating", None)
                if r is None or (isinstance(r, float) and np.isnan(r)):
                    return -1e18
                return float(r)
            return sorted(movies, key=keyfn, reverse=True)[:5]

        agg_df["top_movies"] = agg_df["movies"].apply(top5)

        # 节点集合
        all_persons = set(df["person_a"]).union(set(df["person_b"]))

        # 构建图
        self.G.clear()
        self.G.add_nodes_from(all_persons)

        # -----------------------------
        # 方案A：为每个节点写入多重身份 roles（不破坏旧逻辑：仍提供 primary_role）
        # -----------------------------
        if self.persons_dict is not None:
            node_attrs = {}
            for person in all_persons:
                movies = self.persons_dict.get(person, [])
                roles = [m.get("role", "unknown") for m in movies if m.get("role")]
                role_counts = Counter(roles)

                # 统计 unique movies（避免同一电影多个职位导致 movie_count 被“职位数”放大）
                movie_ids = [m.get("movie_id") for m in movies if m.get("movie_id") is not None]
                unique_movie_count = len(set(movie_ids)) if movie_ids else 0

                ratings = [m.get("rating") for m in movies if m.get("rating") is not None and not pd.isna(m.get("rating"))]
                avg_person_rating = float(np.mean(ratings)) if ratings else np.nan

                roles_set = sorted({r for r in roles if r and r != "unknown"})

                # 主角色：为了兼容你旧的可视化/统计
                primary_role = self._pick_primary_role(roles_set)

                node_attrs[person] = {
                    "roles": roles_set,                 # ✅ 多身份
                    "primary_role": primary_role,       # ✅ 兼容旧逻辑
                    "role_counts": dict(role_counts),   # ✅ 各身份次数
                    "movie_count": unique_movie_count,  # ✅ 参与电影数（去重）
                    "avg_rating": avg_person_rating,    # ✅ 该影人参与电影平均分（可选用）
                }

            nx.set_node_attributes(self.G, node_attrs)

        # 加边
        edges_to_add = []
        for _, row in agg_df.iterrows():
            edges_to_add.append((
                row["person_a"], row["person_b"],
                {
                    "weight": int(row["weight"]),
                    "avg_rating": row["avg_rating"],
                    "movies": row["movies"],               # ✅ 全量
                    "top_movies": row["top_movies"],       # ✅ 仍保留
                    "role_pairs": row["role_pairs"],       # ✅ 新增：这对人的角色组合统计
                }
            ))

        self.G.add_edges_from(edges_to_add)

        logger.info(f"网络构建完成: {self.G.number_of_nodes()} 个节点, {self.G.number_of_edges()} 条边")
        return self.G

    def _pick_primary_role(self, roles_set):
        """兼容旧逻辑：多身份时给一个主角色（优先 director > writer > actor）"""
        if not roles_set:
            return "unknown"
        priority = ["director", "writer", "actor"]
        for r in priority:
            if r in roles_set:
                return r
        return roles_set[0]

    def extract_low_rating_subnetwork(self, threshold=RATING_THRESHOLD):
        """提取低分电影子网络（按边 avg_rating 阈值）"""
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
                    "role_pairs": data.get("role_pairs", {}),
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

    # 兼容旧接口：返回主角色
    def get_node_role(self, node):
        if self.persons_dict is None:
            return self.G.nodes[node].get("primary_role", "unknown") if self.G.has_node(node) else "unknown"

        if self.G.has_node(node) and "primary_role" in self.G.nodes[node]:
            return self.G.nodes[node].get("primary_role", "unknown")

        movies = self.persons_dict.get(node, [])
        roles_set = sorted({m.get("role") for m in movies if m.get("role")})
        return self._pick_primary_role(roles_set)

    # 新接口：返回多身份列表
    def get_node_roles(self, node):
        if self.G.has_node(node) and "roles" in self.G.nodes[node]:
            return self.G.nodes[node].get("roles", [])
        if self.persons_dict is None or node not in self.persons_dict:
            return []
        roles_set = sorted({m.get("role") for m in self.persons_dict[node] if m.get("role") and m.get("role") != "unknown"})
        return roles_set
