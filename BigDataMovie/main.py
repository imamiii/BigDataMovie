# main.py
import pandas as pd
import networkx as nx
import json
import os
import logging
from config import *
from data_loader import DataLoader
from data_processor import DataProcessor
from network_builder import NetworkBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_graph_for_echarts(G: nx.Graph, path: str = "graph.json", top_n: int = 500, builder=None, export_all: bool = False):
    """
    将图导出为 ECharts 可用的 JSON
    
    Args:
        G: 网络图
        path: 输出路径
        top_n: 导出度数最高的前N个节点（仅当export_all=False时有效）
        builder: NetworkBuilder实例，用于获取节点角色和电影信息
        export_all: 是否导出所有节点和边
    """
    if G.number_of_nodes() == 0:
        print("图为空，未导出。")
        return

    # 选出节点
    if export_all:
        # 导出所有节点
        nodes_list = list(G.nodes())
    else:
        # 选出度数最高的 top_n 个节点
        degrees = dict(G.degree())
        nodes_list = sorted(degrees, key=lambda x: degrees[x], reverse=True)[:top_n]

    # 构造节点列表
    nodes = []
    for n in nodes_list:
        role = 'unknown'
        movie_count = 0
        avg_rating = 0
        
        if builder and hasattr(builder, 'get_node_role'):
            role = builder.get_node_role(n)
            
        if builder and hasattr(builder, 'persons_dict') and builder.persons_dict:
            if n in builder.persons_dict:
                movies = builder.persons_dict[n]
                movie_count = len(movies)
                if movie_count > 0:
                    total_rating = sum(movie.get('rating', 0) for movie in movies)
                    avg_rating = total_rating / movie_count
        
        nodes.append({
            "id": n,
            "name": n,
            "value": G.degree(n),
            "role": role,
            "movieCount": movie_count,
            "avgRating": avg_rating
        })

    # 构造边列表 - 过滤自环边
    links = []
    for u, v, data in G.edges(data=True):
        if (u in nodes_list and v in nodes_list) and u != v:
            links.append({
                "source": u,
                "target": v,
                "value": data.get("weight", 1),
                "avgRating": data.get("avg_rating", 0),
                "topMovies": data.get("top_movies", [])
            })

    # 导出JSON
    data = {"nodes": nodes, "links": links}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"已导出网络数据到 {path}，节点数={len(nodes)}，边数={len(links)}")


def main():
    """主程序"""
    logger.info("开始影人合作网络分析项目")

    try:
        # 第一步：加载和清洗数据
        logger.info("=" * 60)
        logger.info("第一步：加载和清洗数据")
        logger.info("=" * 60)

        loader = DataLoader(EXCEL_FILE)
        df = loader.load_data()
        df = loader.rename_columns()
        df = loader.clean_data()
        df = loader.validate_data()
        cleaned_file = loader.save_cleaned_data('data/cleaned_movies.csv')
        logger.info(f"清洗后的数据已保存: {cleaned_file}")

        # 第二步：处理数据，提取影人关系
        logger.info("\n" + "=" * 60)
        logger.info("第二步：提取影人关系")
        logger.info("=" * 60)

        processor = DataProcessor(df)
        persons_dict = processor.extract_all_persons()
        edges_df = processor.build_cooperation_edges()
        edges_df.to_csv('data/cooperation_edges.csv', index=False, encoding='utf-8-sig')
        logger.info(f"合作关系边数据已保存: data/cooperation_edges.csv")

        # 保存统计信息
        person_stats = processor.get_person_stats()
        movie_stats = processor.get_movie_stats()
        person_stats.to_csv('data/person_stats.csv', index=False, encoding='utf-8-sig')
        movie_stats.to_csv('data/movie_stats.csv', index=False, encoding='utf-8-sig')
        logger.info("影人和电影统计信息已保存")

        # 第三步：构建网络
        logger.info("\n" + "=" * 60)
        logger.info("第三步：构建合作网络")
        logger.info("=" * 60)

        builder = NetworkBuilder()
        G = builder.build_network(edges_df, persons_dict)
        G_low_rating = builder.extract_low_rating_subnetwork(RATING_THRESHOLD)

        # 获取高频合作对
        high_coop_pairs = builder.get_high_cooperation_pairs(MIN_COOPERATION_COUNT)
        high_coop_df = pd.DataFrame(high_coop_pairs)
        high_coop_df.to_csv('data/high_cooperation_pairs.csv', index=False, encoding='utf-8-sig')
        logger.info(f"高频合作对已保存: data/high_cooperation_pairs.csv")

        # 计算网络指标和社区检测
        builder.calculate_network_metrics()
        logger.info(f"网络指标计算完成")

        try:
            partition = builder.detect_communities(COMMUNITY_DETECTION_METHOD)
        except Exception as e:
            logger.warning(f"社区检测失败，将不使用社区信息进行可视化: {e}")

        # 第四步：网络分析
        logger.info("\n" + "=" * 60)
        logger.info("第四步：网络分析")
        logger.info("=" * 60)

        try:
            total_nodes = G.number_of_nodes()
            total_edges = G.number_of_edges()
            density = nx.density(G) if total_nodes > 1 else 0
            avg_degree = sum(d for n, d in G.degree()) / total_nodes if total_nodes > 0 else 0
            
            logger.info(f"网络基本结构: 节点数={total_nodes}, 边数={total_edges}, 密度={density:.4f}, 平均度={avg_degree:.2f}")
            
            # 生成分析报告
            report = "影人合作网络分析报告\n" + "="*40 + "\n"
            report += f"网络基本信息:\n"
            report += f"- 总节点数: {total_nodes}\n"
            report += f"- 总边数: {total_edges}\n"
            report += f"- 网络密度: {density:.4f}\n"
            report += f"- 平均度: {avg_degree:.2f}\n"
            
            with open('data/analysis_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"分析报告已保存: data/analysis_report.txt")
            print("\n" + report)
            
        except Exception as e:
            logger.error(f"网络分析过程出错: {e}", exc_info=True)
            print(f"\n❌ 网络分析过程出错: {e}")

        # 第五步：导出ECharts格式数据（用于前端可视化）
        logger.info("\n" + "=" * 60)
        logger.info("第五步：导出ECharts可视化数据")
        logger.info("=" * 60)
        
        try:
            if G_low_rating and G_low_rating.number_of_nodes() > 0:
                # 导出更多节点（500个）以提供更完整的数据
                export_graph_for_echarts(G_low_rating, path="data/graph_top500.json", top_n=500, builder=builder)
                logger.info("✅ 已导出低分电影网络数据到 data/graph_top500.json")
            else:
                logger.warning("⚠️ 低分电影网络为空，跳过导出")
        
            # 同时导出完整网络数据用于展示更多关系
            export_graph_for_echarts(G, path="data/full_network_data.json", top_n=1000, builder=builder)
            logger.info("✅ 已导出完整网络数据到 data/full_network_data.json")
        except Exception as e:
            logger.error(f"❌ 导出ECharts数据失败: {e}", exc_info=True)
            print(f"\n❌ 导出ECharts数据失败: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("项目运行完成！所有结果已保存到 outputs/ 目录")
        logger.info("=" * 60)

        # 打印关键发现
        print("\n✨ 关键发现总结 ✨")
        print("=" * 40)

        # 输出网络密度比较
        if G_low_rating and G_low_rating.number_of_nodes() > 0:
            density_whole = nx.density(G)
            density_low = nx.density(G_low_rating)

            if density_low > density_whole * 1.2:
                print("🔴 重要发现：低分电影网络密度显著高于整体网络")
                print("   这意味着存在紧密的'烂片圈子'，这些影人经常在一起制作低分电影")
            elif density_low > density_whole:
                print("🟡 发现：低分电影网络密度略高于整体网络")
                print("   存在一些小范围的重复合作模式")
            else:
                print("🟢 发现：低分电影网络相对稀疏")
                print("   没有明显的'烂片圈子'现象")

        # 输出前5个高频合作对
        if not high_coop_df.empty:
            print(f"\n🤝 高频合作影人对 (合作次数 ≥ {MIN_COOPERATION_COUNT}):")
            for _, row in high_coop_df.head().iterrows():
                print(f"   {row['person1']} & {row['person2']}: "
                      f"合作{row['cooperation_count']}次, 平均评分{row['avg_rating']:.1f}")

    except Exception as e:
        logger.error(f"程序运行出错: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # 创建必要的目录
    import os

    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    main()