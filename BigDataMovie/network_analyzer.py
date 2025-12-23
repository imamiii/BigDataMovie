# network_analyzer.py
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    def __init__(self, G, G_low_rating=None):
        self.G = G
        self.G_low_rating = G_low_rating

    def analyze_network_structure(self):
        logger.info("开始分析网络结构...")

        analysis_results = {}

        logger.info("计算基本网络指标...")
        analysis_results['basic_metrics'] = self._calculate_basic_metrics()

        logger.info("分析度分布...")
        analysis_results['degree_distribution'] = self._analyze_degree_distribution()

        logger.info("分析聚类特性...")
        analysis_results['clustering_analysis'] = self._analyze_clustering()

        logger.info("分析中心性...")
        analysis_results['centrality_analysis'] = self._analyze_centrality()

        if self.G_low_rating and self.G_low_rating.number_of_nodes() > 0:
            logger.info("对比低分网络与整体网络...")
            analysis_results['low_rating_comparison'] = self._compare_low_rating_network()

        logger.info("网络结构分析完成")
        return analysis_results

    def _calculate_basic_metrics(self):
        metrics = {
            '节点数': self.G.number_of_nodes(),
            '边数': self.G.number_of_edges(),
            '网络密度': nx.density(self.G),
            '平均路径长度': self._calculate_average_path_length(),
            '平均聚类系数': nx.average_clustering(self.G)
        }

        if self.G.number_of_nodes() < 1000 and nx.is_connected(self.G):
            metrics['网络直径'] = nx.diameter(self.G)
        else:
            metrics['网络直径'] = "不连通或计算成本过高"

        return metrics

    def _calculate_average_path_length(self):
        if self.G.number_of_nodes() < 5000:
            if nx.is_connected(self.G):
                return nx.average_shortest_path_length(self.G)
            else:
                largest_component = max(nx.connected_components(self.G), key=len)
                if len(largest_component) < 5000:
                    subgraph = self.G.subgraph(largest_component)
                    return nx.average_shortest_path_length(subgraph)
        return "计算成本过高"

    def _analyze_degree_distribution(self):
        degrees = [d for _, d in self.G.degree()]

        distribution = {
            '平均度': np.mean(degrees) if degrees else 0,
            '最大度': max(degrees) if degrees else 0,
            '最小度': min(degrees) if degrees else 0,
            '度分布': Counter(degrees)
        }

        hist, bins = np.histogram(degrees, bins=20) if degrees else (np.array([]), np.array([]))
        distribution['直方图'] = {'频数': hist.tolist(), '区间': bins.tolist()}

        return distribution

    def _analyze_clustering(self):
        avg_clustering = nx.average_clustering(self.G)

        if self.G.number_of_nodes() < 5000:
            clustering_coeffs = nx.clustering(self.G)
        else:
            logger.info("网络较大，只计算平均聚类系数")
            clustering_coeffs = None

        analysis = {
            '平均聚类系数': avg_clustering,
            '高聚类节点': []
        }

        if clustering_coeffs is not None:
            sorted_nodes = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)[:10]
            for node, coeff in sorted_nodes:
                analysis['高聚类节点'].append({'节点': node, '聚类系数': coeff})

        return analysis

    def _analyze_centrality(self):
        degree_cent = nx.degree_centrality(self.G)

        centrality = {
            '度中心性前10': sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        }

        if self.G.number_of_nodes() < 1000:
            betweenness_cent = nx.betweenness_centrality(self.G, k=min(50, self.G.number_of_nodes()))
            centrality['中介中心性前10'] = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:10]

            closeness_cent = nx.closeness_centrality(self.G)
            centrality['接近中心性前10'] = sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            logger.info("网络较大，跳过中介中心性和接近中心性计算以提高性能")

        return centrality

    def _compare_low_rating_network(self):
        if not self.G_low_rating or self.G_low_rating.number_of_nodes() == 0:
            return None

        comparison = {}

        comparison['节点数对比'] = {
            '整体网络': self.G.number_of_nodes(),
            '低分网络': self.G_low_rating.number_of_nodes(),
            '占比': self.G_low_rating.number_of_nodes() / self.G.number_of_nodes() * 100 if self.G.number_of_nodes() else 0
        }

        comparison['边数对比'] = {
            '整体网络': self.G.number_of_edges(),
            '低分网络': self.G_low_rating.number_of_edges(),
            '占比': self.G_low_rating.number_of_edges() / self.G.number_of_edges() * 100 if self.G.number_of_edges() else 0
        }

        comparison['密度对比'] = {
            '整体网络': nx.density(self.G),
            '低分网络': nx.density(self.G_low_rating)
        }

        comparison['平均聚类系数对比'] = {
            '整体网络': nx.average_clustering(self.G),
            '低分网络': nx.average_clustering(self.G_low_rating)
        }

        low_degree_cent = nx.degree_centrality(self.G_low_rating)
        comparison['低分网络核心节点'] = sorted(low_degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]

        return comparison

    def find_high_risk_clusters(self, min_coop_count=3, min_avg_rating=5.0):
        """找出高风险影人团簇（兼容你的边结构：从 movies / avg_rating 里取评分）"""
        high_risk_clusters = []

        if self.G_low_rating and self.G_low_rating.number_of_nodes() > 0:
            components = list(nx.connected_components(self.G_low_rating))

            for i, component in enumerate(components):
                if len(component) >= 3:
                    subgraph = self.G_low_rating.subgraph(component)

                    cluster_info = {
                        'cluster_id': i + 1,
                        'size': len(component),
                        'density': nx.density(subgraph),
                        'avg_degree': float(np.mean([d for _, d in subgraph.degree()])),
                        'members': list(component)[:10]
                    }

                    # 从边的 movies 里汇总评分（你的边没有 data['ratings']）
                    ratings = []
                    for _, _, data in subgraph.edges(data=True):
                        if "movies" in data and isinstance(data["movies"], list):
                            for m in data["movies"]:
                                r = m.get("rating", None)
                                if r is not None and not pd.isna(r):
                                    ratings.append(float(r))
                        else:
                            r = data.get("avg_rating", None)
                            if r is not None and not pd.isna(r):
                                ratings.append(float(r))

                    if ratings:
                        cluster_info['avg_rating'] = float(np.mean(ratings))
                        cluster_info['rating_count'] = int(len(ratings))

                        if (cluster_info['avg_rating'] < min_avg_rating and
                                len(component) >= 3 and
                                cluster_info['density'] > 0.3):
                            high_risk_clusters.append(cluster_info)

        logger.info(f"找到 {len(high_risk_clusters)} 个高风险影人团簇")
        return high_risk_clusters

    def generate_insights_report(self, analysis_results):
        report = []
        report.append("=" * 60)
        report.append("影人合作网络分析报告")
        report.append("=" * 60)

        metrics = analysis_results['basic_metrics']
        report.append("\n1. 基本网络指标:")
        for key, value in metrics.items():
            report.append(f"   {key}: {value}")

        degree_dist = analysis_results['degree_distribution']
        report.append(f"\n2. 度分布分析:")
        report.append(f"   平均度: {degree_dist['平均度']:.2f}")
        report.append(f"   最大度: {degree_dist['最大度']} (最活跃的影人)")

        centrality = analysis_results['centrality_analysis']
        report.append("\n3. 中心性分析:")
        report.append("   度中心性最高的影人:")
        for i, (person, score) in enumerate(centrality['度中心性前10'][:5], 1):
            report.append(f"     {i}. {person}: {score:.3f}")

        if 'low_rating_comparison' in analysis_results:
            comparison = analysis_results['low_rating_comparison']
            report.append("\n4. 低分电影网络分析:")

            node_comp = comparison['节点数对比']
            density_comp = comparison['密度对比']

            report.append(f"   低分网络包含 {node_comp['低分网络']} 个影人 "
                          f"({node_comp['占比']:.1f}% 的整体网络)")
            report.append(f"   低分网络密度: {density_comp['低分网络']:.4f} "
                          f"vs 整体密度: {density_comp['整体网络']:.4f}")

            if density_comp['低分网络'] > density_comp['整体网络']:
                report.append("   🔍 发现: 低分电影网络密度更高，存在明显的团簇结构！")
            else:
                report.append("   🔍 发现: 低分电影网络相对稀疏")

        high_risk_clusters = self.find_high_risk_clusters()
        if high_risk_clusters:
            report.append("\n5. 高风险影人团簇检测:")
            for cluster in high_risk_clusters[:5]:
                report.append(f"   团簇{cluster['cluster_id']}: "
                              f"{cluster['size']}人, "
                              f"平均评分{cluster.get('avg_rating', 0):.1f}, "
                              f"密度{cluster['density']:.3f}")
                report.append(f"       核心成员: {', '.join(cluster['members'][:3])}...")

        return "\n".join(report)