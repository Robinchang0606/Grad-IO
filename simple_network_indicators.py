#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上市公司网络指标构建工具 - 简化版本
仅使用Python标准库，不依赖pandas、numpy、networkx
"""

import csv
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

class SimpleNetworkIndicatorBuilder:
    """
    简化的网络指标构建器
    """
    
    def __init__(self):
        """
        初始化网络指标构建器
        """
        self.companies = {}  # 公司信息字典
        self.transactions = []  # 交易信息列表
        self.network_graph = defaultdict(list)  # 网络图邻接表
        
    def create_sample_data(self):
        """
        创建示例数据
        """
        # 示例公司数据
        self.companies = {
            'A001': {'name': '公司A1', 'is_listed': 1, 'industry': '制造业', 'assets': 1000000},
            'A002': {'name': '公司A2', 'is_listed': 0, 'industry': '制造业', 'assets': 500000},
            'A003': {'name': '公司A3', 'is_listed': 0, 'industry': '服务业', 'assets': 300000},
            'A004': {'name': '公司A4', 'is_listed': 0, 'industry': '金融业', 'assets': 200000},
            'A005': {'name': '公司A5', 'is_listed': 0, 'industry': '科技业', 'assets': 150000},
            'B001': {'name': '公司B1', 'is_listed': 1, 'industry': '制造业', 'assets': 800000},
            'B002': {'name': '公司B2', 'is_listed': 0, 'industry': '服务业', 'assets': 400000},
            'B003': {'name': '公司B3', 'is_listed': 0, 'industry': '金融业', 'assets': 250000}
        }
        
        # 示例交易数据
        self.transactions = [
            {'id': 1, 'from': 'A001', 'to': 'A002', 'amount': 1000000, 'type': '投资', 'year': 2023},
            {'id': 2, 'from': 'A001', 'to': 'A003', 'amount': 800000, 'type': '采购', 'year': 2023},
            {'id': 3, 'from': 'A001', 'to': 'A004', 'amount': 600000, 'type': '销售', 'year': 2023},
            {'id': 4, 'from': 'A002', 'to': 'A003', 'amount': 300000, 'type': '投资', 'year': 2023},
            {'id': 5, 'from': 'A003', 'to': 'A005', 'amount': 200000, 'type': '服务', 'year': 2023},
            {'id': 6, 'from': 'B001', 'to': 'B002', 'amount': 1200000, 'type': '投资', 'year': 2023},
            {'id': 7, 'from': 'B001', 'to': 'B003', 'amount': 900000, 'type': '采购', 'year': 2023},
            {'id': 8, 'from': 'B002', 'to': 'B003', 'amount': 400000, 'type': '服务', 'year': 2023},
            {'id': 9, 'from': 'A001', 'to': 'A005', 'amount': 150000, 'type': '投资', 'year': 2023},
            {'id': 10, 'from': 'B001', 'to': 'A001', 'amount': 500000, 'type': '合作', 'year': 2023}
        ]
        
        print("示例数据创建成功!")
        print(f"公司数据: {len(self.companies)} 条记录")
        print(f"交易数据: {len(self.transactions)} 条记录")
        
    def build_network_graph(self):
        """
        构建网络图
        """
        # 清空网络图
        self.network_graph = defaultdict(list)
        
        # 构建邻接表
        for transaction in self.transactions:
            from_company = transaction['from']
            to_company = transaction['to']
            
            # 添加边信息
            edge_info = {
                'target': to_company,
                'amount': transaction['amount'],
                'type': transaction['type'],
                'year': transaction['year']
            }
            self.network_graph[from_company].append(edge_info)
            
        print(f"网络图构建完成: {len(self.companies)} 个节点, {len(self.transactions)} 条边")
        
    def calculate_indicator_1_direct_investment(self) -> List[Dict]:
        """
        指标1: 计算上市公司关联公司间的直接投入金额加总
        """
        results = []
        
        # 获取所有上市公司
        listed_companies = [company_id for company_id, info in self.companies.items() 
                          if info['is_listed'] == 1]
        
        for company_id in listed_companies:
            company_info = self.companies[company_id]
            
            # 计算对外直接投资
            outgoing_investment = 0
            investment_count = 0
            
            # 遍历该公司的所有出边
            for edge in self.network_graph[company_id]:
                if edge['type'] in ['投资', '投入']:
                    outgoing_investment += edge['amount']
                    investment_count += 1
            
            # 计算接受的直接投资
            incoming_investment = 0
            
            # 遍历所有交易，找到指向该公司的投资
            for transaction in self.transactions:
                if (transaction['to'] == company_id and 
                    transaction['type'] in ['投资', '投入']):
                    incoming_investment += transaction['amount']
            
            # 总直接投入
            total_direct_investment = outgoing_investment + incoming_investment
            
            results.append({
                'company_id': company_id,
                'company_name': company_info['name'],
                'outgoing_investment': outgoing_investment,
                'incoming_investment': incoming_investment,
                'total_direct_investment': total_direct_investment,
                'investment_transactions_count': investment_count
            })
        
        # 计算投资强度（标准化）
        if results:
            max_investment = max(r['total_direct_investment'] for r in results)
            for result in results:
                if max_investment > 0:
                    result['investment_intensity'] = result['total_direct_investment'] / max_investment
                else:
                    result['investment_intensity'] = 0
        
        return results
    
    def calculate_indicator_2_network_size(self) -> List[Dict]:
        """
        指标2: 计算企业实际拥有的关联交易公司数量
        """
        results = []
        
        # 获取所有上市公司
        listed_companies = [company_id for company_id, info in self.companies.items() 
                          if info['is_listed'] == 1]
        
        for company_id in listed_companies:
            company_info = self.companies[company_id]
            
            # 直接关联公司
            direct_partners = set()
            
            # 出边连接的公司
            for edge in self.network_graph[company_id]:
                direct_partners.add(edge['target'])
            
            # 入边连接的公司
            for transaction in self.transactions:
                if transaction['to'] == company_id:
                    direct_partners.add(transaction['from'])
            
            # 间接关联公司（二度连接）
            indirect_partners = set()
            for partner in direct_partners:
                # 合作伙伴的合作伙伴
                for edge in self.network_graph[partner]:
                    target = edge['target']
                    if target != company_id and target not in direct_partners:
                        indirect_partners.add(target)
                
                # 指向合作伙伴的公司
                for transaction in self.transactions:
                    if transaction['to'] == partner:
                        source = transaction['from']
                        if source != company_id and source not in direct_partners:
                            indirect_partners.add(source)
            
            # 按交易类型分类统计
            investment_partners = set()
            trading_partners = set()
            service_partners = set()
            
            # 统计出边关系
            for edge in self.network_graph[company_id]:
                transaction_type = edge['type']
                target = edge['target']
                
                if '投资' in transaction_type:
                    investment_partners.add(target)
                elif transaction_type in ['采购', '销售']:
                    trading_partners.add(target)
                elif '服务' in transaction_type:
                    service_partners.add(target)
            
            # 统计入边关系
            for transaction in self.transactions:
                if transaction['to'] == company_id:
                    transaction_type = transaction['type']
                    source = transaction['from']
                    
                    if '投资' in transaction_type:
                        investment_partners.add(source)
                    elif transaction_type in ['采购', '销售']:
                        trading_partners.add(source)
                    elif '服务' in transaction_type:
                        service_partners.add(source)
            
            # 计算中心性指标（简化版本）
            total_nodes = len(self.companies)
            degree_centrality = len(direct_partners) / (total_nodes - 1) if total_nodes > 1 else 0
            
            results.append({
                'company_id': company_id,
                'company_name': company_info['name'],
                'direct_partners_count': len(direct_partners),
                'indirect_partners_count': len(indirect_partners),
                'total_network_size': len(direct_partners) + len(indirect_partners),
                'investment_partners': len(investment_partners),
                'trading_partners': len(trading_partners),
                'service_partners': len(service_partners),
                'degree_centrality': degree_centrality
            })
        
        # 计算网络密度
        if results:
            max_network_size = max(r['total_network_size'] for r in results)
            for result in results:
                if max_network_size > 0:
                    result['network_density'] = result['total_network_size'] / max_network_size
                else:
                    result['network_density'] = 0
        
        return results
    
    def generate_network_report(self) -> Dict:
        """
        生成综合网络分析报告
        """
        # 计算两个指标
        indicator_1 = self.calculate_indicator_1_direct_investment()
        indicator_2 = self.calculate_indicator_2_network_size()
        
        # 合并两个指标
        combined_analysis = []
        
        # 创建指标2的字典，便于查找
        indicator_2_dict = {item['company_id']: item for item in indicator_2}
        
        for item1 in indicator_1:
            company_id = item1['company_id']
            combined_item = item1.copy()
            
            # 添加指标2的数据
            if company_id in indicator_2_dict:
                item2 = indicator_2_dict[company_id]
                for key, value in item2.items():
                    if key not in combined_item:
                        combined_item[key] = value
            
            combined_analysis.append(combined_item)
        
        # 计算综合评分
        if combined_analysis:
            max_investment = max(item['total_direct_investment'] for item in combined_analysis)
            max_network_size = max(item['total_network_size'] for item in combined_analysis)
            
            for item in combined_analysis:
                # 标准化评分
                investment_score = (item['total_direct_investment'] / max_investment) if max_investment > 0 else 0
                network_score = (item['total_network_size'] / max_network_size) if max_network_size > 0 else 0
                
                item['investment_score'] = investment_score
                item['network_score'] = network_score
                
                # 综合网络影响力评分
                item['network_influence_score'] = (
                    0.4 * investment_score + 
                    0.3 * network_score + 
                    0.3 * item['degree_centrality']
                )
        
        # 计算汇总统计
        total_companies = len(combined_analysis)
        avg_investment = sum(item['total_direct_investment'] for item in combined_analysis) / total_companies if total_companies > 0 else 0
        avg_network_size = sum(item['total_network_size'] for item in combined_analysis) / total_companies if total_companies > 0 else 0
        max_influence = max(item['network_influence_score'] for item in combined_analysis) if combined_analysis else 0
        
        report = {
            'indicator_1_data': indicator_1,
            'indicator_2_data': indicator_2,
            'combined_analysis': combined_analysis,
            'summary_stats': {
                'total_listed_companies': total_companies,
                'avg_direct_investment': avg_investment,
                'avg_network_size': avg_network_size,
                'max_network_influence': max_influence
            }
        }
        
        return report
    
    def export_results_to_csv(self, report: Dict, output_dir: str = "/workspace"):
        """
        导出结果到CSV文件
        """
        # 导出指标1结果
        indicator_1_file = f"{output_dir}/indicator_1_direct_investment.csv"
        self._write_csv(indicator_1_file, report['indicator_1_data'])
        
        # 导出指标2结果
        indicator_2_file = f"{output_dir}/indicator_2_network_size.csv"
        self._write_csv(indicator_2_file, report['indicator_2_data'])
        
        # 导出综合分析结果
        combined_file = f"{output_dir}/combined_network_analysis.csv"
        self._write_csv(combined_file, report['combined_analysis'])
        
        print(f"结果已导出到:")
        print(f"- 指标1 (直接投入): {indicator_1_file}")
        print(f"- 指标2 (网络规模): {indicator_2_file}")
        print(f"- 综合分析: {combined_file}")
        
    def _write_csv(self, filename: str, data: List[Dict]):
        """
        写入CSV文件
        """
        if not data:
            return
            
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    def print_table(self, data: List[Dict], title: str, max_cols: int = 6):
        """
        打印表格格式的数据
        """
        if not data:
            print(f"{title}: 无数据")
            return
            
        print(f"\n{title}")
        print("=" * len(title))
        
        # 获取要显示的列
        all_keys = list(data[0].keys())
        display_keys = all_keys[:max_cols]
        
        # 打印表头
        header = " | ".join(f"{key:>12}" for key in display_keys)
        print(header)
        print("-" * len(header))
        
        # 打印数据行
        for row in data:
            row_str = " | ".join(f"{str(row.get(key, ''))[:12]:>12}" for key in display_keys)
            print(row_str)

def main():
    """
    主函数 - 演示网络指标构建过程
    """
    print("=" * 60)
    print("上市公司网络指标构建工具 - 简化版")
    print("=" * 60)
    
    # 初始化构建器
    builder = SimpleNetworkIndicatorBuilder()
    
    # 创建示例数据
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
    builder.print_table(
        report['indicator_1_data'], 
        "指标1结果",
        max_cols=6
    )
    
    print("\n" + "="*50)
    print("指标2: 关联交易公司数量")
    print("="*50)
    builder.print_table(
        report['indicator_2_data'], 
        "指标2结果",
        max_cols=6
    )
    
    print("\n" + "="*50)
    print("综合网络分析")
    print("="*50)
    # 选择关键列显示
    combined_display = []
    for item in report['combined_analysis']:
        combined_display.append({
            'company_name': item['company_name'],
            'total_investment': item['total_direct_investment'],
            'network_size': item['total_network_size'],
            'influence_score': round(item['network_influence_score'], 3)
        })
    
    builder.print_table(combined_display, "综合分析结果")
    
    print("\n" + "="*50)
    print("汇总统计")
    print("="*50)
    stats = report['summary_stats']
    print(f"上市公司总数: {stats['total_listed_companies']}")
    print(f"平均直接投入: {stats['avg_direct_investment']:,.2f}")
    print(f"平均网络规模: {stats['avg_network_size']:.2f}")
    print(f"最高网络影响力: {stats['max_network_influence']:.3f}")
    
    # 导出结果
    builder.export_results_to_csv(report)
    
    print("\n网络指标构建完成！")

if __name__ == "__main__":
    main()