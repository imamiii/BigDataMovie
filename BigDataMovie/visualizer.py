# visualizer.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors
import logging
from config import NODE_SIZE_MULTIPLIER, EDGE_WIDTH_MULTIPLIER, MAX_NODES_TO_LABEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkVisualizer:
    def __init__(self, figsize=(20, 16)):
        self.figsize = figsize
        # 设置中文显示字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def visualize_network(self, G, partition=None, highlight_nodes=None,
                          title="影人合作网络", filename="network.png"):
        """可视化网络"""
        num_nodes = G.number_of_nodes()
        logger.info(f"开始可视化网络，原始大小: {num_nodes}个节点, {G.number_of_edges()}条边")
        
        # 对于非常大的网络，只可视化核心子网络（度大于等于2的节点）
        if num_nodes > 5000:
            logger.info("网络过大，只可视化核心子网络（度≥2的节点）...")
            # 只保留度≥2的节点
            core_nodes = [node for node, degree in G.degree() if degree >= 2]
            # 创建子图
            G = G.subgraph(core_nodes).copy()
            num_nodes = G.number_of_nodes()
            logger.info(f"核心子网络大小: {num_nodes}个节点, {G.number_of_edges()}条边")
            
            # 如果子网络仍然过大（>10000个节点），进一步筛选
            if num_nodes > 10000:
                logger.info("子网络仍然过大，只保留度≥3的节点...")
                core_nodes = [node for node, degree in G.degree() if degree >= 3]
                G = G.subgraph(core_nodes).copy()
                num_nodes = G.number_of_nodes()
                logger.info(f"进一步筛选后的子网络大小: {num_nodes}个节点, {G.number_of_edges()}条边")
        
        if num_nodes == 0:
            logger.warning("没有足够的节点用于可视化")
            return
            
        plt.figure(figsize=self.figsize)

        # 计算节点大小（基于度中心性）
        degrees = dict(G.degree())
        node_sizes = [degrees[node] * NODE_SIZE_MULTIPLIER + 100 for node in G.nodes()]

        # 计算边宽度（基于权重）
        edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
        edge_widths = [w * EDGE_WIDTH_MULTIPLIER for w in edge_weights]

        # 节点颜色（基于社区或度）
        if partition and num_nodes <= 1000:
            # 只对小型网络使用社区分区着色（计算量大）
            communities = set(partition.values())
            colors = cm.rainbow(np.linspace(0, 1, len(communities)))
            node_colors = [colors[partition[node]] for node in G.nodes()]
        else:
            # 使用度中心性着色
            node_colors = [degrees[node] for node in G.nodes()]

        # 选择布局算法 - 优化节点间距和布局质量
        logger.info(f"开始布局计算，网络大小: {num_nodes}个节点, {G.number_of_edges()}条边")
        
        if num_nodes < 300:
            # 小网络使用spring_layout，增加节点间距
            pos = nx.spring_layout(G, k=2 / np.sqrt(num_nodes), iterations=50, seed=42)
        elif num_nodes < 1000:
            # 中等网络使用kamada_kawai_layout
            pos = nx.kamada_kawai_layout(G)
        elif num_nodes < 3000:
            # 较大网络使用spring_layout，增加迭代次数和节点间距
            pos = nx.spring_layout(G, k=3 / np.sqrt(num_nodes), iterations=30, seed=42)
        else:
            # 大型网络使用更适合的布局，增加节点间距
            pos = nx.spring_layout(G, k=4 / np.sqrt(num_nodes), iterations=20, seed=42)
        
        # 重新缩放布局以增加节点间距
        pos = nx.rescale_layout_dict(pos, scale=2.0)
        
        logger.info("布局计算完成，开始绘制网络...")

        # 绘制网络
        nodes = nx.draw_networkx_nodes(G, pos,
                                       node_size=node_sizes,
                                       node_color=node_colors,
                                       cmap=plt.cm.viridis if not partition else None,
                                       alpha=0.8)

        # 绘制边
        edges = nx.draw_networkx_edges(G, pos,
                                       width=edge_widths,
                                       alpha=0.3,
                                       edge_color='gray')

        # 突出显示特定节点
        if highlight_nodes:
            # 只保留在当前子图中的节点
            highlight_nodes = [node for node in highlight_nodes if node in G]
            if highlight_nodes:
                highlight_sizes = [degrees[node] * NODE_SIZE_MULTIPLIER + 200 for node in highlight_nodes]
                nx.draw_networkx_nodes(G, pos,
                                       nodelist=highlight_nodes,
                                       node_size=highlight_sizes,
                                       node_color='red',
                                       alpha=0.9)

        # 添加标签（只显示非常重要的节点）
        if num_nodes <= 1000:  # 只有小网络才显示标签
            if num_nodes <= 200:
                # 小网络显示所有标签
                nx.draw_networkx_labels(G, pos, font_size=8, font_family='SimHei', font_weight='bold')
            else:
                # 只显示高度中心性的节点标签
                high_degree_nodes = [node for node, degree in degrees.items()
                                     if degree > np.percentile(list(degrees.values()), 95)]  # 只显示前5%，减少重叠
                labels = {node: node for node in high_degree_nodes}
                nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='SimHei', font_weight='bold')

        plt.title(title, fontsize=20, fontweight='bold', font_family='SimHei')
        plt.axis('off')

        # 添加图例
        if partition and num_nodes <= 1000:
            self._add_community_legend(G, partition, colors)

        # 保存图片（进一步降低分辨率以提高速度）
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"网络图已保存到: {filename}")
        plt.close()  # 关闭图形以释放内存
        logger.info("网络可视化完成")

    def _add_community_legend(self, G, partition, colors):
        """添加社区图例"""
        from matplotlib.patches import Patch

        # 统计每个社区的大小
        community_sizes = {}
        for node, comm_id in partition.items():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1

        # 创建图例
        legend_elements = []
        for comm_id, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:8]:  # 只显示前8个
            legend_elements.append(Patch(facecolor=colors[comm_id],
                                         label=f'社区{comm_id + 1}: {size}人',
                                         alpha=0.8))

        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    def plot_degree_distribution(self, G, filename="degree_distribution.png"):
        """绘制度分布图"""
        degrees = [d for n, d in G.degree()]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 度分布直方图
        axes[0].hist(degrees, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('度', fontsize=12)
        axes[0].set_ylabel('频数', fontsize=12)
        axes[0].set_title('度分布直方图', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 度分布对数坐标
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        axes[1].loglog(unique_degrees, counts, 'bo', alpha=0.6)
        axes[1].set_xlabel('度 (对数坐标)', fontsize=12)
        axes[1].set_ylabel('频数 (对数坐标)', fontsize=12)
        axes[1].set_title('度分布（双对数坐标）', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'网络度分布分析 (平均度: {np.mean(degrees):.2f})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"度分布图已保存到: {filename}")
        plt.close()  # 关闭图形以释放内存

    def plot_network_metrics_comparison(self, G, G_low_rating, filename="network_comparison.png"):
        """绘制网络指标对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 节点数对比
        axes[0, 0].bar(['整体网络', '低分网络'],
                       [G.number_of_nodes(),
                        G_low_rating.number_of_nodes() if G_low_rating else 0],
                       color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('节点数对比', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('节点数')

        # 边数对比
        axes[0, 1].bar(['整体网络', '低分网络'],
                       [G.number_of_edges(),
                        G_low_rating.number_of_edges() if G_low_rating else 0],
                       color=['skyblue', 'lightcoral'])
        axes[0, 1].set_title('边数对比', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('边数')

        # 网络密度对比
        axes[0, 2].bar(['整体网络', '低分网络'],
                       [nx.density(G),
                        nx.density(G_low_rating) if G_low_rating and G_low_rating.number_of_nodes() > 0 else 0],
                       color=['skyblue', 'lightcoral'])
        axes[0, 2].set_title('网络密度对比', fontsize=12, fontweight='bold')
        axes[0, 2].set_ylabel('密度')

        # 平均聚类系数对比
        axes[1, 0].bar(['整体网络', '低分网络'],
                       [nx.average_clustering(G),
                        nx.average_clustering(
                            G_low_rating) if G_low_rating and G_low_rating.number_of_nodes() > 0 else 0],
                       color=['skyblue', 'lightcoral'])
        axes[1, 0].set_title('平均聚类系数对比', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('平均聚类系数')

        # 平均度对比
        avg_degree_G = np.mean([d for n, d in G.degree()])
        avg_degree_low = np.mean(
            [d for n, d in G_low_rating.degree()]) if G_low_rating and G_low_rating.number_of_nodes() > 0 else 0
        axes[1, 1].bar(['整体网络', '低分网络'],
                       [avg_degree_G, avg_degree_low],
                       color=['skyblue', 'lightcoral'])
        axes[1, 1].set_title('平均度对比', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('平均度')

        # 连通分量数量对比
        components_G = nx.number_connected_components(G)
        components_low = nx.number_connected_components(G_low_rating) if G_low_rating else 0
        axes[1, 2].bar(['整体网络', '低分网络'],
                       [components_G, components_low],
                       color=['skyblue', 'lightcoral'])
        axes[1, 2].set_title('连通分量数量对比', fontsize=12, fontweight='bold')
        axes[1, 2].set_ylabel('连通分量数')

        plt.suptitle('网络结构指标对比分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"网络对比图已保存到: {filename}")
        plt.close()  # 关闭图形以释放内存

    def plot_high_risk_clusters(self, high_risk_clusters, filename="high_risk_clusters.png"):
        """绘制高风险团簇图"""
        if not high_risk_clusters:
            logger.warning("没有高风险团簇可绘制")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 团簇大小分布
        sizes = [cluster['size'] for cluster in high_risk_clusters]
        densities = [cluster['density'] for cluster in high_risk_clusters]
        avg_ratings = [cluster.get('avg_rating', 0) for cluster in high_risk_clusters]

        # 散点图：团簇大小 vs 密度
        scatter1 = axes[0].scatter(sizes, densities, c=avg_ratings,
                                   cmap='RdYlGn_r', s=100, alpha=0.7, edgecolors='black')
        axes[0].set_xlabel('团簇大小（人数）', fontsize=12)
        axes[0].set_ylabel('团簇密度', fontsize=12)
        axes[0].set_title('高风险团簇：大小 vs 密度', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='平均评分')

        # 前10大团簇
        top_clusters = sorted(high_risk_clusters, key=lambda x: x['size'], reverse=True)[:10]
        cluster_ids = [f"团簇{cluster['cluster_id']}" for cluster in top_clusters]
        cluster_sizes = [cluster['size'] for cluster in top_clusters]
        cluster_ratings = [cluster.get('avg_rating', 0) for cluster in top_clusters]

        bars = axes[1].bar(cluster_ids, cluster_sizes,
                           color=plt.cm.RdYlGn_r(np.array(cluster_ratings) / 10))
        axes[1].set_xlabel('团簇ID', fontsize=12)
        axes[1].set_ylabel('团簇大小（人数）', fontsize=12)
        axes[1].set_title('前10大高风险团簇', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)

        # 添加评分标签
        for i, (bar, rating) in enumerate(zip(bars, cluster_ratings)):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{rating:.1f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('高风险影人团簇分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"高风险团簇图已保存到: {filename}")
        plt.show()

    def create_summary_dashboard(self, G, G_low_rating, analysis_results,
                                 high_risk_clusters, filename="summary_dashboard.png"):
        """创建分析总结仪表板"""
        fig = plt.figure(figsize=(20, 12))

        # 创建子图布局
        gs = fig.add_gridspec(3, 3)

        # 1. 网络概览（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['节点数', '边数', '平均度', '网络密度']
        values = [
            G.number_of_nodes(),
            G.number_of_edges(),
            np.mean([d for n, d in G.degree()]),
            nx.density(G)
        ]
        bars1 = ax1.barh(metrics, values, color=plt.cm.Set3(np.arange(len(metrics)) / len(metrics)))
        ax1.set_title('整体网络概览', fontsize=12, fontweight='bold')
        ax1.set_xlabel('数值')

        # 2. 低分网络概览（中上）
        ax2 = fig.add_subplot(gs[0, 1])
        if G_low_rating and G_low_rating.number_of_nodes() > 0:
            metrics_low = ['节点数', '边数', '平均度', '网络密度']
            values_low = [
                G_low_rating.number_of_nodes(),
                G_low_rating.number_of_edges(),
                np.mean([d for n, d in G_low_rating.degree()]),
                nx.density(G_low_rating)
            ]
            bars2 = ax2.barh(metrics_low, values_low, color=plt.cm.Set3(np.arange(len(metrics_low)) / len(metrics_low)))
        ax2.set_title('低分网络概览', fontsize=12, fontweight='bold')
        ax2.set_xlabel('数值')

        # 3. 对比分析（右上）
        ax3 = fig.add_subplot(gs[0, 2])
        comparison_metrics = ['密度对比', '聚类系数对比']
        if G_low_rating and G_low_rating.number_of_nodes() > 0:
            whole_density = nx.density(G)
            low_density = nx.density(G_low_rating)
            whole_clustering = nx.average_clustering(G)
            low_clustering = nx.average_clustering(G_low_rating)

            x = np.arange(len(comparison_metrics))
            width = 0.35
            ax3.bar(x - width / 2, [whole_density, whole_clustering], width, label='整体网络', color='skyblue')
            ax3.bar(x + width / 2, [low_density, low_clustering], width, label='低分网络', color='lightcoral')
            ax3.set_xticks(x)
            ax3.set_xticklabels(comparison_metrics)
            ax3.legend()
        ax3.set_title('网络指标对比', fontsize=12, fontweight='bold')
        ax3.set_ylabel('数值')

        # 4. 高风险团簇（左下，跨两行）
        ax4 = fig.add_subplot(gs[1:, 0])
        if high_risk_clusters:
            cluster_ids = [f"C{cluster['cluster_id']}" for cluster in high_risk_clusters[:8]]
            cluster_sizes = [cluster['size'] for cluster in high_risk_clusters[:8]]
            cluster_ratings = [cluster.get('avg_rating', 0) for cluster in high_risk_clusters[:8]]

            bars4 = ax4.bar(cluster_ids, cluster_sizes,
                            color=plt.cm.RdYlGn_r(np.array(cluster_ratings) / 10))
            ax4.set_xlabel('团簇ID')
            ax4.set_ylabel('团簇大小（人数）')
            ax4.set_title('高风险团簇分析', fontsize=12, fontweight='bold')

            # 添加评分标签
            for i, (bar, rating) in enumerate(zip(bars4, cluster_ratings)):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{rating:.1f}', ha='center', va='bottom', fontsize=9)

        # 5. 度分布（中下）
        ax5 = fig.add_subplot(gs[1, 1])
        degrees = [d for n, d in G.degree()]
        ax5.hist(degrees, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_xlabel('度')
        ax5.set_ylabel('频数')
        ax5.set_title('度分布直方图', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. 中心性分析（右下）
        ax6 = fig.add_subplot(gs[1, 2])
        if 'centrality_analysis' in analysis_results:
            centrality = analysis_results['centrality_analysis']
            top_nodes = [node for node, _ in centrality['度中心性前10'][:5]]
            top_scores = [score for _, score in centrality['度中心性前10'][:5]]
            bars6 = ax6.barh(top_nodes, top_scores, color=plt.cm.Pastel1(np.arange(5) / 5))
            ax6.set_xlabel('度中心性')
            ax6.set_title('度中心性最高的影人', fontsize=12, fontweight='bold')

        # 7. 洞察总结（下中）
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.axis('off')

        # 生成总结文本
        insights = self._generate_insights_text(G, G_low_rating, analysis_results, high_risk_clusters)
        ax7.text(0.05, 0.95, insights, transform=ax7.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('影人合作网络分析仪表板', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"分析仪表板已保存到: {filename}")
        plt.show()

    def _generate_insights_text(self, G, G_low_rating, analysis_results, high_risk_clusters):
        """生成洞察总结文本"""
        insights = []

        # 基本洞察
        insights.append("📊 网络分析关键洞察:")
        insights.append(f"• 网络包含 {G.number_of_nodes()} 位影人，{G.number_of_edges()} 次合作")

        if G_low_rating and G_low_rating.number_of_nodes() > 0:
            insights.append(f"• 低分电影网络包含 {G_low_rating.number_of_nodes()} 位影人 "
                            f"({G_low_rating.number_of_nodes() / G.number_of_nodes() * 100:.1f}% 的整体网络)")

            # 密度比较
            whole_density = nx.density(G)
            low_density = nx.density(G_low_rating)
            if low_density > whole_density * 1.2:
                insights.append("• 🔴 低分网络密度显著更高，存在明显的'烂片圈子'")
            elif low_density > whole_density:
                insights.append("• 🟡 低分网络密度略高，可能存在小范围重复合作")
            else:
                insights.append("• 🟢 低分网络密度较低，无明显的烂片聚集现象")

        # 高风险团簇洞察
        if high_risk_clusters:
            insights.append(f"\n⚠️ 高风险影人团簇检测:")
            insights.append(f"• 发现 {len(high_risk_clusters)} 个高风险团簇")

            largest_cluster = max(high_risk_clusters, key=lambda x: x['size'])
            insights.append(f"• 最大团簇包含 {largest_cluster['size']} 人，平均评分 {largest_cluster.get('avg_rating', 0):.1f}")

            if len(high_risk_clusters) >= 3:
                insights.append("• 多个高风险团簇同时存在，可能存在系统性合作问题")

        # 中心性洞察
        if 'centrality_analysis' in analysis_results:
            centrality = analysis_results['centrality_analysis']
            top_person, top_score = centrality['度中心性前10'][0]
            insights.append(f"\n🏆 核心影人: {top_person} (度中心性: {top_score:.3f})")

        return '\n'.join(insights)