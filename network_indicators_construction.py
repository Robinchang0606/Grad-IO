#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上市公司网络指标构建工具
构建两个核心网络指标：
1. 关联公司间直接投入金额加总
2. 企业实际拥有的关联交易公司数量
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class NetworkIndicatorBuilder:
    """
    上市公司网络指标构建器
    """
    
    def __init__(self):
        """
        初始化网络指标构建器
        """
        self.company_data = None
        self.transaction_data = None
        self.network_graph = None
        
    def load_data(self, company_file: str, transaction_file: str):
        """
        加载上市公司关联方基本信息和交易数据
        
        Parameters:
        -----------
        company_file : str
            公司基本信息文件路径
        transaction_file : str  
            关联交易信息文件路径
        """
        try:
            self.company_data = pd.read_csv(company_file)
            self.transaction_data = pd.read_csv(transaction_file)
            print("数据加载成功!")
            print(f"公司数据: {len(self.company_data)} 条记录")
            print(f"交易数据: {len(self.transaction_data)} 条记录")
        except Exception as e:
            print(f"数据加载失败: {e}")
            
    def create_sample_data(self):
        """
        创建示例数据用于演示
        """
        # 创建示例公司数据
        company_data = {
            'company_id': ['A001', 'A002', 'A003', 'A004', 'A005', 'B001', 'B002', 'B003'],
            'company_name': ['公司A1', '公司A2', '公司A3', '公司A4', '公司A5', '公司B1', '公司B2', '公司B3'],
            'is_listed': [1, 0, 0, 0, 0, 1, 0, 0],  # 1表示上市公司
            'industry': ['制造业', '制造业', '服务业', '金融业', '科技业', '制造业', '服务业', '金融业'],
            'total_assets': [1000000, 500000, 300000, 200000, 150000, 800000, 400000, 250000]
        }
        
        # 创建示例关联交易数据
        transaction_data = {
            'transaction_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'from_company': ['A001', 'A001', 'A001', 'A002', 'A003', 'B001', 'B001', 'B002', 'A001', 'B001'],
            'to_company': ['A002', 'A003', 'A004', 'A003', 'A005', 'B002', 'B003', 'B003', 'A005', 'A001'],
            'transaction_amount': [1000000, 800000, 600000, 300000, 200000, 1200000, 900000, 400000, 150000, 500000],
            'transaction_type': ['投资', '采购', '销售', '投资', '服务', '投资', '采购', '服务', '投资', '合作'],
            'year': [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]
        }
        
        self.company_data = pd.DataFrame(company_data)
        self.transaction_data = pd.DataFrame(transaction_data)
        
        print("示例数据创建成功!")
        print(f"公司数据: {len(self.company_data)} 条记录")
        print(f"交易数据: {len(self.transaction_data)} 条记录")
        
    def build_network_graph(self):
        """
        构建网络图
        """
        if self.transaction_data is None:
            raise ValueError("请先加载交易数据")
            
        # 创建有向图
        self.network_graph = nx.DiGraph()
        
        # 添加节点（公司）
        for _, company in self.company_data.iterrows():
            self.network_graph.add_node(
                company['company_id'],
                name=company['company_name'],
                is_listed=company['is_listed'],
                industry=company['industry'],
                assets=company['total_assets']
            )
        
        # 添加边（交易关系）
        for _, transaction in self.transaction_data.iterrows():
            self.network_graph.add_edge(
                transaction['from_company'],
                transaction['to_company'],
                amount=transaction['transaction_amount'],
                type=transaction['transaction_type'],
                year=transaction['year']
            )
            
        print(f"网络图构建完成: {self.network_graph.number_of_nodes()} 个节点, {self.network_graph.number_of_edges()} 条边")
        
    def calculate_indicator_1_direct_investment(self) -> pd.DataFrame:
        """
        指标1: 计算上市公司关联公司间的直接投入金额加总
        
        Returns:
        --------
        pd.DataFrame
            包含每个上市公司的直接投入金额加总
        """
        if self.network_graph is None:
            self.build_network_graph()
            
        results = []
        
        # 获取所有上市公司
        listed_companies = [node for node in self.network_graph.nodes() 
                          if self.network_graph.nodes[node]['is_listed'] == 1]
        
        for company in listed_companies:
            # 计算该公司作为投资方的直接投入总额
            outgoing_investment = 0
            investment_count = 0
            
            # 遍历该公司的所有出边（对外投资）
            for _, target, edge_data in self.network_graph.out_edges(company, data=True):
                if edge_data.get('type') in ['投资', '投入']:  # 只考虑投资类型的交易
                    outgoing_investment += edge_data.get('amount', 0)
                    investment_count += 1
            
            # 计算该公司接受的直接投入总额
            incoming_investment = 0
            
            # 遍历该公司的所有入边（接受投资）
            for source, _, edge_data in self.network_graph.in_edges(company, data=True):
                if edge_data.get('type') in ['投资', '投入']:
                    incoming_investment += edge_data.get('amount', 0)
            
            # 计算总的直接投入（对外投资 + 接受投资）
            total_direct_investment = outgoing_investment + incoming_investment
            
            results.append({
                'company_id': company,
                'company_name': self.network_graph.nodes[company]['name'],
                'outgoing_investment': outgoing_investment,  # 对外直接投资
                'incoming_investment': incoming_investment,  # 接受直接投资
                'total_direct_investment': total_direct_investment,  # 总直接投入
                'investment_transactions_count': investment_count  # 投资交易笔数
            })
        
        indicator_1_df = pd.DataFrame(results)
        
        # 添加相对指标（标准化）
        if len(indicator_1_df) > 0:
            indicator_1_df['investment_intensity'] = (
                indicator_1_df['total_direct_investment'] / 
                indicator_1_df['total_direct_investment'].max()
            )
        
        return indicator_1_df
    
    def calculate_indicator_2_network_size(self) -> pd.DataFrame:
        """
        指标2: 计算企业实际拥有的关联交易公司数量
        
        Returns:
        --------
        pd.DataFrame
            包含每个上市公司的关联交易公司数量指标
        """
        if self.network_graph is None:
            self.build_network_graph()
            
        results = []
        
        # 获取所有上市公司
        listed_companies = [node for node in self.network_graph.nodes() 
                          if self.network_graph.nodes[node]['is_listed'] == 1]
        
        for company in listed_companies:
            # 直接关联公司数量（一度连接）
            direct_partners = set()
            
            # 出边连接的公司
            for _, target in self.network_graph.out_edges(company):
                direct_partners.add(target)
            
            # 入边连接的公司
            for source, _ in self.network_graph.in_edges(company):
                direct_partners.add(source)
            
            # 间接关联公司数量（二度连接）
            indirect_partners = set()
            for partner in direct_partners:
                # 合作伙伴的合作伙伴
                for _, target in self.network_graph.out_edges(partner):
                    if target != company and target not in direct_partners:
                        indirect_partners.add(target)
                for source, _ in self.network_graph.in_edges(partner):
                    if source != company and source not in direct_partners:
                        indirect_partners.add(source)
            
            # 按交易类型分类统计
            investment_partners = set()  # 投资关系伙伴
            trading_partners = set()     # 交易关系伙伴
            service_partners = set()     # 服务关系伙伴
            
            # 统计不同类型的关联关系
            for _, target, edge_data in self.network_graph.out_edges(company, data=True):
                transaction_type = edge_data.get('type', '')
                if '投资' in transaction_type:
                    investment_partners.add(target)
                elif transaction_type in ['采购', '销售']:
                    trading_partners.add(target)
                elif '服务' in transaction_type:
                    service_partners.add(target)
            
            for source, _, edge_data in self.network_graph.in_edges(company, data=True):
                transaction_type = edge_data.get('type', '')
                if '投资' in transaction_type:
                    investment_partners.add(source)
                elif transaction_type in ['采购', '销售']:
                    trading_partners.add(source)
                elif '服务' in transaction_type:
                    service_partners.add(source)
            
            # 计算网络中心性指标
            degree_centrality = nx.degree_centrality(self.network_graph)[company]
            betweenness_centrality = nx.betweenness_centrality(self.network_graph)[company]
            closeness_centrality = nx.closeness_centrality(self.network_graph)[company]
            
            results.append({
                'company_id': company,
                'company_name': self.network_graph.nodes[company]['name'],
                'direct_partners_count': len(direct_partners),      # 直接关联公司数量
                'indirect_partners_count': len(indirect_partners),  # 间接关联公司数量
                'total_network_size': len(direct_partners) + len(indirect_partners),  # 总网络规模
                'investment_partners': len(investment_partners),    # 投资伙伴数量
                'trading_partners': len(trading_partners),          # 交易伙伴数量
                'service_partners': len(service_partners),          # 服务伙伴数量
                'degree_centrality': degree_centrality,             # 度中心性
                'betweenness_centrality': betweenness_centrality,   # 介数中心性
                'closeness_centrality': closeness_centrality       # 接近中心性
            })
        
        indicator_2_df = pd.DataFrame(results)
        
        # 添加相对指标（标准化）
        if len(indicator_2_df) > 0:
            indicator_2_df['network_density'] = (
                indicator_2_df['total_network_size'] / 
                (len(self.network_graph.nodes()) - 1)  # 除以可能的最大连接数
            )
        
        return indicator_2_df
    
    def generate_network_report(self) -> Dict:
        """
        生成综合网络分析报告
        
        Returns:
        --------
        Dict
            包含两个指标和综合分析的报告
        """
        # 计算两个指标
        indicator_1 = self.calculate_indicator_1_direct_investment()
        indicator_2 = self.calculate_indicator_2_network_size()
        
        # 合并两个指标
        combined_df = pd.merge(
            indicator_1, 
            indicator_2, 
            on=['company_id', 'company_name'], 
            how='outer'
        )
        
        # 计算综合网络指标
        if len(combined_df) > 0:
            # 标准化各项指标
            combined_df['investment_score'] = (
                combined_df['total_direct_investment'] / 
                combined_df['total_direct_investment'].max()
            ) if combined_df['total_direct_investment'].max() > 0 else 0
            
            combined_df['network_score'] = (
                combined_df['total_network_size'] / 
                combined_df['total_network_size'].max()
            ) if combined_df['total_network_size'].max() > 0 else 0
            
            # 综合网络影响力评分
            combined_df['network_influence_score'] = (
                0.4 * combined_df['investment_score'] + 
                0.3 * combined_df['network_score'] + 
                0.3 * combined_df['degree_centrality']
            )
        
        # 生成报告
        report = {
            'indicator_1_data': indicator_1,
            'indicator_2_data': indicator_2,
            'combined_analysis': combined_df,
            'summary_stats': {
                'total_listed_companies': len(combined_df),
                'avg_direct_investment': combined_df['total_direct_investment'].mean() if len(combined_df) > 0 else 0,
                'avg_network_size': combined_df['total_network_size'].mean() if len(combined_df) > 0 else 0,
                'max_network_influence': combined_df['network_influence_score'].max() if len(combined_df) > 0 else 0
            }
        }
        
        return report
    
    def export_results(self, report: Dict, output_dir: str = "/workspace"):
        """
        导出分析结果
        
        Parameters:
        -----------
        report : Dict
            网络分析报告
        output_dir : str
            输出目录
        """
        # 导出指标1结果
        indicator_1_file = f"{output_dir}/indicator_1_direct_investment.csv"
        report['indicator_1_data'].to_csv(indicator_1_file, index=False, encoding='utf-8-sig')
        
        # 导出指标2结果
        indicator_2_file = f"{output_dir}/indicator_2_network_size.csv"
        report['indicator_2_data'].to_csv(indicator_2_file, index=False, encoding='utf-8-sig')
        
        # 导出综合分析结果
        combined_file = f"{output_dir}/combined_network_analysis.csv"
        report['combined_analysis'].to_csv(combined_file, index=False, encoding='utf-8-sig')
        
        print(f"结果已导出到:")
        print(f"- 指标1 (直接投入): {indicator_1_file}")
        print(f"- 指标2 (网络规模): {indicator_2_file}")
        print(f"- 综合分析: {combined_file}")

def main():
    """
    主函数 - 演示网络指标构建过程
    """
    print("=" * 60)
    print("上市公司网络指标构建工具")
    print("=" * 60)
    
    # 初始化构建器
    builder = NetworkIndicatorBuilder()
    
    # 创建示例数据（实际使用时替换为真实数据加载）
    builder.create_sample_data()
    
    # 构建网络图
    builder.build_network_graph()
    
    # 生成网络分析报告
    print("\n正在计算网络指标...")
    report = builder.generate_network_report()
    
    # 显示结果
    print("\n" + "="*50)
    print("指标1: 关联公司间直接投入金额")
    print("="*50)
    print(report['indicator_1_data'].to_string(index=False))
    
    print("\n" + "="*50)
    print("指标2: 关联交易公司数量")
    print("="*50)
    print(report['indicator_2_data'].to_string(index=False))
    
    print("\n" + "="*50)
    print("综合网络分析")
    print("="*50)
    key_columns = ['company_name', 'total_direct_investment', 'total_network_size', 'network_influence_score']
    print(report['combined_analysis'][key_columns].to_string(index=False))
    
    print("\n" + "="*50)
    print("汇总统计")
    print("="*50)
    for key, value in report['summary_stats'].items():
        print(f"{key}: {value:,.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")
    
    # 导出结果
    builder.export_results(report)
    
    print("\n网络指标构建完成！")

if __name__ == "__main__":
    main()