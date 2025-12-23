# main.py
import numpy as np
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


def export_graph_for_echarts(
    G,
    builder: NetworkBuilder,
    person_stats_df: pd.DataFrame,
    path="data/full_network_data.json",
    top_n=1000,
    export_all=False
):
    if G.number_of_nodes() == 0:
        raise ValueError("图为空，无法导出")

    ps = None
    if person_stats_df is not None and len(person_stats_df) > 0:
        ps = person_stats_df.set_index("name")

    # 选节点
    if export_all:
        nodes_list = list(G.nodes())
    else:
        degrees = dict(G.degree())
        nodes_list = sorted(degrees, key=lambda x: degrees[x], reverse=True)[:top_n]

    node_set = set(nodes_list)

    nodes = []
    for n in nodes_list:
        # === 方案A：多身份 ===
        # primary_role（兼容旧逻辑）
        primary_role = "unknown"
        roles = []
        role_counts = {}

        if G.has_node(n):
            primary_role = G.nodes[n].get("primary_role", "unknown")
            roles = G.nodes[n].get("roles", []) or []
            role_counts = G.nodes[n].get("role_counts", {}) or {}

        # movieCount：建议用 persons_dict 的 unique movie_id 去重
        movie_count = 0
        if builder and builder.persons_dict and n in builder.persons_dict:
            movie_ids = [m.get("movie_id") for m in builder.persons_dict[n] if m.get("movie_id") is not None]
            movie_count = len(set(movie_ids))

        avg_rating = np.nan
        if ps is not None and n in ps.index:
            avg_rating = ps.loc[n, "avg_rating"]

        nodes.append({
            "id": n,
            "name": n,
            "value": int(G.degree(n)),

            # ✅ 兼容旧字段：role = primary_role（用于着色/图例）
            "role": primary_role,

            # ✅ 新增：多身份
            "primaryRole": primary_role,
            "roles": roles,                 # e.g. ["director","actor"]
            "roleCounts": role_counts,      # e.g. {"director":3,"actor":5}

            "movieCount": int(movie_count),
            "avgRating": None if pd.isna(avg_rating) else float(avg_rating)
        })

    links = []
    for u, v, data in G.edges(data=True):
        if u not in node_set or v not in node_set:
            continue

        weight = int(data.get("weight", 1))
        avg_rating = data.get("avg_rating", np.nan)

        movies = data.get("movies", [])
        top_movies = data.get("top_movies", [])

        role_pairs = data.get("role_pairs", {})  # ✅ 方案A新增（可选给前端展示）

        links.append({
            "source": u,
            "target": v,
            "value": weight,
            "avgRating": None if pd.isna(avg_rating) else float(avg_rating),
            "movies": movies,
            "topMovies": top_movies,

            # ✅ 新增：这对人的角色组合统计
            "rolePairs": role_pairs
        })

    out = {"nodes": nodes, "links": links}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"导出完成: nodes={len(nodes)}, links={len(links)} -> {path}")


def main():
    logger.info("开始影人合作网络分析项目")

    try:
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

        logger.info("\n" + "=" * 60)
        logger.info("第二步：提取影人关系")
        logger.info("=" * 60)

        processor = DataProcessor(df)
        persons_dict = processor.extract_all_persons()
        edges_df = processor.build_cooperation_edges()
        edges_df.to_csv('data/cooperation_edges.csv', index=False, encoding='utf-8-sig')
        logger.info("合作关系边数据已保存: data/cooperation_edges.csv")

        person_stats = processor.get_person_stats()
        movie_stats = processor.get_movie_stats()
        person_stats.to_csv('data/person_stats.csv', index=False, encoding='utf-8-sig')
        movie_stats.to_csv('data/movie_stats.csv', index=False, encoding='utf-8-sig')
        logger.info("影人和电影统计信息已保存")

        logger.info("\n" + "=" * 60)
        logger.info("第三步：构建合作网络")
        logger.info("=" * 60)

        builder = NetworkBuilder()
        G = builder.build_network(edges_df, persons_dict)
        G_low_rating = builder.extract_low_rating_subnetwork(RATING_THRESHOLD)

        high_coop_pairs = builder.get_high_cooperation_pairs(MIN_COOPERATION_COUNT)
        high_coop_df = pd.DataFrame(high_coop_pairs)
        high_coop_df.to_csv('data/high_cooperation_pairs.csv', index=False, encoding='utf-8-sig')
        logger.info("高频合作对已保存: data/high_cooperation_pairs.csv")

        builder.calculate_network_metrics()
        logger.info("网络指标计算完成")

        try:
            builder.detect_communities(COMMUNITY_DETECTION_METHOD)
        except Exception as e:
            logger.warning(f"社区检测失败，将不使用社区信息: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("第四步：网络分析")
        logger.info("=" * 60)

        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        density = nx.density(G) if total_nodes > 1 else 0
        avg_degree = sum(d for _, d in G.degree()) / total_nodes if total_nodes > 0 else 0

        report = "影人合作网络分析报告\n" + "=" * 40 + "\n"
        report += f"网络基本信息:\n"
        report += f"- 总节点数: {total_nodes}\n"
        report += f"- 总边数: {total_edges}\n"
        report += f"- 网络密度: {density:.4f}\n"
        report += f"- 平均度: {avg_degree:.2f}\n"

        with open('data/analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info("分析报告已保存: data/analysis_report.txt")

        logger.info("\n" + "=" * 60)
        logger.info("第五步：导出ECharts可视化数据")
        logger.info("=" * 60)

        if G_low_rating and G_low_rating.number_of_nodes() > 0:
            export_graph_for_echarts(
                G_low_rating, builder, person_stats,
                path="data/graph_top500.json",
                top_n=500
            )
            logger.info("✅ 已导出低分电影网络数据到 data/graph_top500.json")
        else:
            logger.warning("⚠️ 低分电影网络为空，跳过导出")

        export_graph_for_echarts(
            G, builder, person_stats,
            path="data/full_network_data.json",
            top_n=1000
        )
        logger.info("✅ 已导出完整网络数据到 data/full_network_data.json")

        os.makedirs("static/data", exist_ok=True)
        with open("data/full_network_data.json", "r", encoding="utf-8") as rf:
            out = json.load(rf)
        with open("static/data/network_data.json", "w", encoding="utf-8") as wf:
            json.dump(out, wf, ensure_ascii=False)
        logger.info("✅ 同步一份到 static/data/network_data.json")

        logger.info("\n" + "=" * 60)
        logger.info("项目运行完成！")
        logger.info("=" * 60)

        print("\n" + report)

    except Exception as e:
        logger.error(f"程序运行出错: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    main()
