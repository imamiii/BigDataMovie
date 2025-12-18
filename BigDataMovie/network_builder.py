# network_builder.py
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from config import RATING_THRESHOLD, MIN_COOPERATION_COUNT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkBuilder:
    def __init__(self):
        self.G = nx.Graph()  # 整体网络
        self.G_low_rating = None  # 低分电影子网络
        self.edge_data = defaultdict(list)  # 边数据
        self.persons_dict = None  # 影人信息字典

    def build_network(self, edges_df, persons_dict=None):
        """从边数据构建网络"""
        logger.info("开始构建影人合作网络...")
        logger.info(f"边数据大小: {len(edges_df)} 条记录")
        
        # 保存影人信息字典
        if persons_dict is not None:
            self.persons_dict = persons_dict

        # 优化：使用更高效的方式处理边数据
        logger.info("正在聚合边数据...")
        
        # 按person1和person2分组，计算基本统计信息
        grouped_edges = edges_df.groupby(['person1', 'person2']).agg({
            'rating': ['mean', 'size'],  # 平均评分和合作次数
            'movie_name': 'size',  # 只是为了保持分组
            'year': 'size'  # 只是为了保持分组
        })
        
        grouped_edges.columns = ['avg_rating', 'weight', '_', '_']
        grouped_edges = grouped_edges.drop(['_', '_'], axis=1).reset_index()
        
        total_edges = len(grouped_edges)
        logger.info(f"聚合完成，共有 {total_edges} 条边")

        # 获取所有影人
        all_persons = set(edges_df['person1']).union(set(edges_df['person2']))

        # 清空图
        self.G.clear()

        # 添加节点
        num_nodes = len(all_persons)
        logger.info(f"添加 {num_nodes} 个节点...")
        self.G.add_nodes_from(all_persons)

        # 准备边数据（批量添加优化）
        logger.info("正在准备边数据...")
        edges_to_add = []
        edge_count = 0
        
        # 为了高效获取top movies，创建一个临时字典
        edge_key_to_movies = defaultdict(list)
        
        # 先将所有电影数据按person1和person2分组，过滤掉自己与自己的合作
        for _, row in edges_df.iterrows():
            if row['person1'] == row['person2']:
                continue  # 跳过自己与自己的合作
            key = (row['person1'], row['person2'])
            edge_key_to_movies[key].append({
                'movie_name': row['movie_name'],
                'rating': row['rating'],
                'year': row['year']
            })
        
        for _, row in grouped_edges.iterrows():
            person1 = row['person1']
            person2 = row['person2']
            weight = row['weight']
            avg_rating = row['avg_rating']
            
            # 获取评分最高的前5部电影
            movies = edge_key_to_movies.get((person1, person2), [])
            # 按评分排序，取前5部
            top_movies = sorted(movies, key=lambda x: x['rating'], reverse=True)[:5]
            
            # 准备边数据
            edges_to_add.append((person1, person2, {
                'weight': weight,
                'avg_rating': avg_rating,
                'top_movies': top_movies
            }))
            edge_count += 1
            
            # 每处理10000条边记录一条日志
            if edge_count % 10000 == 0:
                logger.info(f"已准备 {edge_count}/{total_edges} 条边...")

        # 批量添加边（提高效率）
        logger.info(f"开始批量添加 {edge_count} 条边...")
        self.G.add_edges_from(edges_to_add)

        logger.info(f"网络构建完成: {self.G.number_of_nodes()} 个节点, {edge_count} 条边")
        return self.G

    def extract_low_rating_subnetwork(self, threshold=RATING_THRESHOLD):
        """提取低分电影子网络"""
        logger.info(f"提取低分电影子网络 (阈值={threshold})...")

        # 找出平均评分低于阈值的边
        low_rating_edges = []
        for u, v, data in self.G.edges(data=True):
            avg_rating = data['avg_rating']
            if not pd.isna(avg_rating) and avg_rating < threshold:
                low_rating_edges.append((u, v))

        # 创建子网络
        if low_rating_edges:
            self.G_low_rating = self.G.edge_subgraph(low_rating_edges)
            logger.info(f"低分电影子网络: {self.G_low_rating.number_of_nodes()} 个节点, "
                        f"{self.G_low_rating.number_of_edges()} 条边")
        else:
            self.G_low_rating = nx.Graph()
            logger.warning("未找到低分电影边")

        return self.G_low_rating

    def get_high_cooperation_pairs(self, min_count=MIN_COOPERATION_COUNT):
        """获取高频合作影人对"""
        high_coop_pairs = []

        for u, v, data in self.G.edges(data=True):
            if data['weight'] >= min_count:
                # 使用直接存储的平均评分
                avg_rating = data['avg_rating']

                pair_info = {
                    'person1': u,
                    'person2': v,
                    'cooperation_count': data['weight'],
                    'avg_rating': avg_rating,
                    'movies': data['top_movies'],  # 使用已存储的前5部高分电影
                    'movies_count': len(data['top_movies'])
                }
                high_coop_pairs.append(pair_info)

        # 按合作次数排序
        high_coop_pairs.sort(key=lambda x: x['cooperation_count'], reverse=True)
        logger.info(f"找到 {len(high_coop_pairs)} 对合作次数 ≥ {min_count} 的影人组合")

        return high_coop_pairs

    def calculate_network_metrics(self):
        """计算网络指标"""
        metrics = {}

        # 整体网络指标
        metrics['total_nodes'] = self.G.number_of_nodes()
        metrics['total_edges'] = self.G.number_of_edges()
        metrics['density'] = nx.density(self.G)
        metrics['avg_clustering'] = nx.average_clustering(self.G)

        # 计算连通分量
        connected_components = list(nx.connected_components(self.G))
        metrics['connected_components'] = len(connected_components)
        metrics['largest_component_size'] = max(len(c) for c in connected_components) if connected_components else 0

        # 度分布统计
        degrees = [d for n, d in self.G.degree()]
        metrics['avg_degree'] = np.mean(degrees) if degrees else 0
        metrics['max_degree'] = max(degrees) if degrees else 0

        # 中心性节点
        degree_centrality = nx.degree_centrality(self.G)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        metrics['top_central_nodes'] = top_nodes

        # 如果存在低分网络，计算其指标
        if self.G_low_rating and self.G_low_rating.number_of_nodes() > 0:
            metrics['low_rating_nodes'] = self.G_low_rating.number_of_nodes()
            metrics['low_rating_edges'] = self.G_low_rating.number_of_edges()
            metrics['low_rating_density'] = nx.density(self.G_low_rating)
            metrics['low_rating_avg_clustering'] = nx.average_clustering(self.G_low_rating)

        return metrics

    def detect_communities(self, method='louvain'):
        """检测社区/团簇"""
        try:
            if method == 'louvain':
                import community as community_louvain
                partition = community_louvain.best_partition(self.G)
                community_count = len(set(partition.values()))
                logger.info(f"Louvain算法检测到 {community_count} 个社区")
                return partition

            elif method == 'girvan_newman':
                # Girvan-Newman算法（较慢，适用于小网络）
                comp = nx.community.girvan_newman(self.G)
                communities = tuple(sorted(c) for c in next(comp))
                community_count = len(communities)
                logger.info(f"Girvan-Newman算法检测到 {community_count} 个社区")
                return communities

            else:
                logger.warning(f"不支持的社区检测方法: {method}")
                return None

        except ImportError as e:
            logger.error(f"社区检测失败，请安装相应库: {e}")
            return None

    def get_node_role(self, node):
        """获取节点的角色信息（director/actor）"""
        if self.persons_dict is None or node not in self.persons_dict:
            return 'unknown'
        
        # 检查影人是否有导演角色
        for movie in self.persons_dict[node]:
            if movie['role'] == 'director':
                return 'director'
        
        # 如果没有导演角色，检查是否有演员角色
        for movie in self.persons_dict[node]:
            if movie['role'] == 'actor':
                return 'actor'
        
        return 'unknown'