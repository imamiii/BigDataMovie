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
        """分析网络结构特征"""
        logger.info("开始分析网络结构...")

        analysis_results = {}

        # 1. 基本网络指标
        logger.info("计算基本网络指标...")
        analysis_results['basic_metrics'] = self._calculate_basic_metrics()

        # 2. 度分布分析
        logger.info("分析度分布...")
        analysis_results['degree_distribution'] = self._analyze_degree_distribution()

        # 3. 聚类系数分析
        logger.info("分析聚类特性...")
        analysis_results['clustering_analysis'] = self._analyze_clustering()

        # 4. 中心性分析
        logger.info("分析中心性...")
        analysis_results['centrality_analysis'] = self._analyze_centrality()

        # 5. 低分网络对比分析
        if self.G_low_rating and self.G_low_rating.number_of_nodes() > 0:
            logger.info("对比低分网络与整体网络...")
            analysis_results['low_rating_comparison'] = self._compare_low_rating_network()

        logger.info("网络结构分析完成")
        return analysis_results

    def _calculate_basic_metrics(self):
        """计算基本网络指标"""
        metrics = {
            '节点数': self.G.number_of_nodes(),
            '边数': self.G.number_of_edges(),
            '网络密度': nx.density(self.G),
            '平均路径长度': self._calculate_average_path_length(),
            '平均聚类系数': nx.average_clustering(self.G)
        }
        
        # 只对小网络计算网络直径（大型网络计算成本太高）
        if self.G.number_of_nodes() < 1000 and nx.is_connected(self.G):
            metrics['网络直径'] = nx.diameter(self.G)
        else:
            metrics['网络直径'] = "不连通或计算成本过高"
        
        return metrics

    def _calculate_average_path_length(self):
        """计算平均路径长度"""
        # 只对小网络计算平均路径长度（大型网络计算成本太高）
        if self.G.number_of_nodes() < 5000:
            if nx.is_connected(self.G):
                return nx.average_shortest_path_length(self.G)
            else:
                # 计算最大连通分量的平均路径长度
                largest_component = max(nx.connected_components(self.G), key=len)
                if len(largest_component) < 5000:
                    subgraph = self.G.subgraph(largest_component)
                    return nx.average_shortest_path_length(subgraph)
        return "计算成本过高"

    def _analyze_degree_distribution(self):
        """分析度分布"""
        degrees = [d for n, d in self.G.degree()]

        distribution = {
            '平均度': np.mean(degrees),
            '最大度': max(degrees),
            '最小度': min(degrees),
            '度分布': Counter(degrees)
        }

        # 计算度分布直方图
        hist, bins = np.histogram(degrees, bins=20)
        distribution['直方图'] = {'频数': hist.tolist(), '区间': bins.tolist()}

        return distribution

    def _analyze_clustering(self):
        """分析聚类特性"""
        # 计算平均聚类系数（比计算所有节点的聚类系数更高效）
        avg_clustering = nx.average_clustering(self.G)
        
        # 只对小网络计算所有节点的聚类系数
        if self.G.number_of_nodes() < 5000:
            clustering_coeffs = nx.clustering(self.G)
        else:
            logger.info("网络较大，只计算平均聚类系数")
            clustering_coeffs = None

        analysis = {
            '平均聚类系数': avg_clustering,
            '高聚类节点': []
        }

        # 找出聚类系数最高的节点（只对小网络计算）
        if clustering_coeffs is not None:
            sorted_nodes = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)[:10]
            for node, coeff in sorted_nodes:
                analysis['高聚类节点'].append({'节点': node, '聚类系数': coeff})

        return analysis

    def _analyze_centrality(self):
        """分析中心性"""
        # 度中心性（快速计算）
        degree_cent = nx.degree_centrality(self.G)
        
        centrality = {
            '度中心性前10': sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        # 只对小网络计算更复杂的中心性指标
        if self.G.number_of_nodes() < 1000:
            # 中介中心性（使用更少的采样点）
            betweenness_cent = nx.betweenness_centrality(self.G, k=min(50, self.G.number_of_nodes()))
            centrality['中介中心性前10'] = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # 接近中心性
            closeness_cent = nx.closeness_centrality(self.G)
            centrality['接近中心性前10'] = sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            logger.info("网络较大，跳过中介中心性和接近中心性计算以提高性能")

        return centrality

    def _compare_low_rating_network(self):
        """比较低分网络与整体网络"""
        if not self.G_low_rating or self.G_low_rating.number_of_nodes() == 0:
            return None

        comparison = {}

        # 基本指标对比
        comparison['节点数对比'] = {
            '整体网络': self.G.number_of_nodes(),
            '低分网络': self.G_low_rating.number_of_nodes(),
            '占比': self.G_low_rating.number_of_nodes() / self.G.number_of_nodes() * 100
        }

        comparison['边数对比'] = {
            '整体网络': self.G.number_of_edges(),
            '低分网络': self.G_low_rating.number_of_edges(),
            '占比': self.G_low_rating.number_of_edges() / self.G.number_of_edges() * 100
        }

        comparison['密度对比'] = {
            '整体网络': nx.density(self.G),
            '低分网络': nx.density(self.G_low_rating)
        }

        comparison['平均聚类系数对比'] = {
            '整体网络': nx.average_clustering(self.G),
            '低分网络': nx.average_clustering(self.G_low_rating)
        }

        # 找出低分网络中的核心节点
        low_degree_cent = nx.degree_centrality(self.G_low_rating)
        comparison['低分网络核心节点'] = sorted(low_degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]

        return comparison

    def find_high_risk_clusters(self, min_coop_count=3, min_avg_rating=5.0):
        """找出高风险影人团簇"""
        high_risk_clusters = []

        # 获取所有连通分量
        if self.G_low_rating and self.G_low_rating.number_of_nodes() > 0:
            components = list(nx.connected_components(self.G_low_rating))

            for i, component in enumerate(components):
                if len(component) >= 3:  # 只考虑3人以上的团簇
                    subgraph = self.G_low_rating.subgraph(component)

                    # 计算团簇指标
                    cluster_info = {
                        'cluster_id': i + 1,
                        'size': len(component),
                        'density': nx.density(subgraph),
                        'avg_degree': np.mean([d for n, d in subgraph.degree()]),
                        'members': list(component)[:10]  # 只显示前10个成员
                    }

                    # 计算平均评分
                    ratings = []
                    for u, v, data in subgraph.edges(data=True):
                        ratings.extend([r for r in data['ratings'] if not pd.isna(r)])

                    if ratings:
                        cluster_info['avg_rating'] = np.mean(ratings)
                        cluster_info['rating_count'] = len(ratings)

                        # 检查是否为高风险团簇
                        if (cluster_info['avg_rating'] < min_avg_rating and
                                len(component) >= 3 and
                                cluster_info['density'] > 0.3):
                            high_risk_clusters.append(cluster_info)

        logger.info(f"找到 {len(high_risk_clusters)} 个高风险影人团簇")
        return high_risk_clusters

    def generate_insights_report(self, analysis_results):
        """生成分析洞察报告"""
        report = []
        report.append("=" * 60)
        report.append("影人合作网络分析报告")
        report.append("=" * 60)

        # 基本指标
        metrics = analysis_results['basic_metrics']
        report.append("\n1. 基本网络指标:")
        for key, value in metrics.items():
            report.append(f"   {key}: {value}")

        # 度分布
        degree_dist = analysis_results['degree_distribution']
        report.append(f"\n2. 度分布分析:")
        report.append(f"   平均度: {degree_dist['平均度']:.2f}")
        report.append(f"   最大度: {degree_dist['最大度']} (最活跃的影人)")

        # 中心性分析
        centrality = analysis_results['centrality_analysis']
        report.append("\n3. 中心性分析:")
        report.append("   度中心性最高的影人:")
        for i, (person, score) in enumerate(centrality['度中心性前10'][:5], 1):
            report.append(f"     {i}. {person}: {score:.3f}")

        # 低分网络对比
        if 'low_rating_comparison' in analysis_results:
            comparison = analysis_results['low_rating_comparison']
            report.append("\n4. 低分电影网络分析:")

            node_comp = comparison['节点数对比']
            edge_comp = comparison['边数对比']
            density_comp = comparison['密度对比']

            report.append(f"   低分网络包含 {node_comp['低分网络']} 个影人 "
                          f"({node_comp['占比']:.1f}% 的整体网络)")
            report.append(f"   低分网络密度: {density_comp['低分网络']:.4f} "
                          f"vs 整体密度: {density_comp['整体网络']:.4f}")

            if density_comp['低分网络'] > density_comp['整体网络']:
                report.append("   🔍 发现: 低分电影网络密度更高，存在明显的团簇结构！")
            else:
                report.append("   🔍 发现: 低分电影网络相对稀疏")

        # 高风险团簇
        high_risk_clusters = self.find_high_risk_clusters()
        if high_risk_clusters:
            report.append("\n5. 高风险影人团簇检测:")
            for cluster in high_risk_clusters[:5]:  # 只显示前5个
                report.append(f"   团簇{cluster['cluster_id']}: "
                              f"{cluster['size']}人, "
                              f"平均评分{cluster.get('avg_rating', 0):.1f}, "
                              f"密度{cluster['density']:.3f}")
                report.append(f"       核心成员: {', '.join(cluster['members'][:3])}...")

        return "\n".join(report)