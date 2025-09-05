#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上市公司关联方网络分析工具
基于关联公司交易信息计算网络指标
"""

import csv
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime

class RelatedPartyNetworkAnalyzer:
    """
    关联方网络分析器
    基于实际的关联方交易数据结构
    """
    
    def __init__(self):
        """
        初始化分析器
        """
        self.related_party_data = []  # 关联方数据
        self.companies = {}  # 公司信息
        self.relation_weights = self._init_relation_weights()
        
    def _init_relation_weights(self) -> Dict[str, float]:
        """
        初始化关联关系权重
        根据关联关系的重要程度设置权重
        """
        return {
            '01': 1.0,   # 上市公司的母公司 - 最高权重
            '02': 0.9,   # 上市公司的子公司 - 高权重
            '03': 0.8,   # 与上市公司受同一母公司控制的其他企业
            '04': 0.9,   # 对上市公司实施共同控制的投资方
            '05': 0.8,   # 对上市公司施加重大影响的投资方
            '06': 0.7,   # 上市公司的合营企业
            '07': 0.7,   # 上市公司的联营企业
            '08': 0.6,   # 上市公司的主要投资者个人及与其关系密切的家庭成员
            '09': 0.5,   # 上市公司或其母公司的关键管理人员及其关系密切的家庭成员
            '10': 0.6,   # 关键管理人员控制的企业
            '11': 0.4,   # 上市公司的关联方之间
            '12': 0.3    # 其他
        }
    
    def load_dta_data(self, file_path: str):
        """
        加载DTA格式的关联方数据
        
        Parameters:
        -----------
        file_path : str
            DTA文件路径
        """
        try:
            # 这里假设DTA文件已经转换为CSV格式
            # 实际使用时可以用pandas.read_stata()读取DTA文件
            with open(file_path, 'r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file)
                self.related_party_data = list(reader)
            
            print(f"数据加载成功! 共 {len(self.related_party_data)} 条关联方记录")
            self._build_company_registry()
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """
        创建示例数据（基于实际DTA结构）
        """
        sample_data = [
            {
                'Stkcd': '000001',
                'Repttype': '1',
                'Reptdt': '2023-12-31',
                'Repart': '深圳投资控股有限公司',
                'Relation': '01',
                'Annodt': '2024-04-15',
                'Relation1': '母公司',
                'Rigicy': '51.2',
                'Cogicy': '0',
                'Regcap': '1000000',
                'Curtype': 'CNY',
                'Corebs': '投资管理',
                'Site': '深圳市',
                'Notes': '',
                'RalatedPartyID': 'RP001'
            },
            {
                'Stkcd': '000001',
                'Repttype': '1',
                'Reptdt': '2023-12-31',
                'Repart': '平安科技有限公司',
                'Relation': '02',
                'Annodt': '2024-04-15',
                'Relation1': '子公司',
                'Rigicy': '0',
                'Cogicy': '100',
                'Regcap': '500000',
                'Curtype': 'CNY',
                'Corebs': '科技服务',
                'Site': '深圳市',
                'Notes': '',
                'RalatedPartyID': 'RP002'
            },
            {
                'Stkcd': '000001',
                'Repttype': '1',
                'Reptdt': '2023-12-31',
                'Repart': '平安银行股份有限公司',
                'Relation': '03',
                'Annodt': '2024-04-15',
                'Relation1': '同一控制下企业',
                'Rigicy': '0',
                'Cogicy': '0',
                'Regcap': '2000000',
                'Curtype': 'CNY',
                'Corebs': '银行业务',
                'Site': '深圳市',
                'Notes': '',
                'RalatedPartyID': 'RP003'
            },
            {
                'Stkcd': '000002',
                'Repttype': '1',
                'Reptdt': '2023-12-31',
                'Repart': '万科企业股份有限公司',
                'Relation': '01',
                'Annodt': '2024-04-20',
                'Relation1': '母公司',
                'Rigicy': '45.8',
                'Cogicy': '0',
                'Regcap': '800000',
                'Curtype': 'CNY',
                'Corebs': '房地产开发',
                'Site': '深圳市',
                'Notes': '',
                'RalatedPartyID': 'RP004'
            },
            {
                'Stkcd': '000002',
                'Repttype': '1',
                'Reptdt': '2023-12-31',
                'Repart': '万科物业发展有限公司',
                'Relation': '02',
                'Annodt': '2024-04-20',
                'Relation1': '子公司',
                'Rigicy': '0',
                'Cogicy': '60',
                'Regcap': '300000',
                'Curtype': 'CNY',
                'Corebs': '物业管理',
                'Site': '深圳市',
                'Notes': '',
                'RalatedPartyID': 'RP005'
            }
        ]
        
        self.related_party_data = sample_data
        print("示例数据创建成功!")
        print(f"共 {len(self.related_party_data)} 条关联方记录")
        self._build_company_registry()
    
    def _build_company_registry(self):
        """
        构建公司注册表
        """
        self.companies = {}
        
        # 从关联方数据中提取公司信息
        for record in self.related_party_data:
            stkcd = record['Stkcd']
            if stkcd not in self.companies:
                self.companies[stkcd] = {
                    'stock_code': stkcd,
                    'is_listed': True,
                    'related_parties': [],
                    'latest_report_date': record['Reptdt']
                }
            
            # 添加关联方信息
            related_party = {
                'name': record['Repart'],
                'relation_code': record['Relation'],
                'relation_desc': record['Relation1'],
                'control_by_related': float(record['Rigicy']) if record['Rigicy'] else 0.0,
                'control_to_related': float(record['Cogicy']) if record['Cogicy'] else 0.0,
                'registered_capital': float(record['Regcap']) if record['Regcap'] else 0.0,
                'currency': record['Curtype'],
                'business': record['Corebs'],
                'location': record['Site'],
                'party_id': record['RalatedPartyID'],
                'report_date': record['Reptdt'],
                'announce_date': record['Annodt']
            }
            
            self.companies[stkcd]['related_parties'].append(related_party)
        
        print(f"构建完成: {len(self.companies)} 家上市公司的关联方网络")
    
    def calculate_network_indicator_1(self, stock_code: str = None) -> List[Dict]:
        """
        指标1: 关联公司间的直接投入/控制权益加总
        
        Parameters:
        -----------
        stock_code : str, optional
            特定股票代码，如果不提供则计算所有公司
            
        Returns:
        --------
        List[Dict]
            关联投入指标结果
        """
        results = []
        
        companies_to_analyze = [stock_code] if stock_code else list(self.companies.keys())
        
        for stkcd in companies_to_analyze:
            if stkcd not in self.companies:
                continue
                
            company_info = self.companies[stkcd]
            related_parties = company_info['related_parties']
            
            # 计算各类投入指标
            total_control_received = 0  # 接受的控制权益
            total_control_given = 0     # 给出的控制权益
            total_registered_capital = 0 # 关联方注册资本总和
            weighted_control_score = 0   # 加权控制评分
            
            # 按关联关系类型分类统计
            relation_stats = defaultdict(list)
            
            for party in related_parties:
                control_received = party['control_by_related']
                control_given = party['control_to_related']
                reg_capital = party['registered_capital']
                relation_code = party['relation_code']
                
                total_control_received += control_received
                total_control_given += control_given
                total_registered_capital += reg_capital
                
                # 计算加权控制评分
                relation_weight = self.relation_weights.get(relation_code, 0.3)
                control_impact = max(control_received, control_given)
                weighted_control_score += control_impact * relation_weight
                
                # 按关系类型分类
                relation_stats[relation_code].append({
                    'name': party['name'],
                    'control_received': control_received,
                    'control_given': control_given,
                    'capital': reg_capital
                })
            
            # 计算衍生指标
            net_control_position = total_control_given - total_control_received
            control_diversity = len(relation_stats)  # 关联关系类型多样性
            avg_party_capital = total_registered_capital / len(related_parties) if related_parties else 0
            
            results.append({
                'stock_code': stkcd,
                'total_related_parties': len(related_parties),
                'total_control_received': total_control_received,
                'total_control_given': total_control_given,
                'net_control_position': net_control_position,
                'total_registered_capital': total_registered_capital,
                'weighted_control_score': weighted_control_score,
                'control_diversity': control_diversity,
                'avg_party_capital': avg_party_capital,
                'latest_report_date': company_info['latest_report_date'],
                'relation_breakdown': dict(relation_stats)
            })
        
        return results
    
    def calculate_network_indicator_2(self, stock_code: str = None) -> List[Dict]:
        """
        指标2: 企业关联交易网络规模和复杂度
        
        Parameters:
        -----------
        stock_code : str, optional
            特定股票代码，如果不提供则计算所有公司
            
        Returns:
        --------
        List[Dict]
            网络规模指标结果
        """
        results = []
        
        companies_to_analyze = [stock_code] if stock_code else list(self.companies.keys())
        
        for stkcd in companies_to_analyze:
            if stkcd not in self.companies:
                continue
                
            company_info = self.companies[stkcd]
            related_parties = company_info['related_parties']
            
            # 计算网络规模指标
            total_parties = len(related_parties)
            
            # 按关联关系分类统计
            parent_companies = len([p for p in related_parties if p['relation_code'] == '01'])
            subsidiaries = len([p for p in related_parties if p['relation_code'] == '02'])
            sister_companies = len([p for p in related_parties if p['relation_code'] == '03'])
            joint_ventures = len([p for p in related_parties if p['relation_code'] in ['06', '07']])
            individual_related = len([p for p in related_parties if p['relation_code'] in ['08', '09', '10']])
            other_related = len([p for p in related_parties if p['relation_code'] in ['11', '12']])
            
            # 计算控制层级
            controlling_parties = len([p for p in related_parties if p['control_by_related'] > 0])
            controlled_parties = len([p for p in related_parties if p['control_to_related'] > 0])
            
            # 计算地理分布
            locations = set(p['location'] for p in related_parties if p['location'])
            geographic_diversity = len(locations)
            
            # 计算业务多样性
            businesses = set(p['business'] for p in related_parties if p['business'])
            business_diversity = len(businesses)
            
            # 计算网络复杂度评分
            complexity_score = (
                0.3 * (total_parties / 10) +  # 规模因子
                0.2 * (len(set(p['relation_code'] for p in related_parties)) / 12) +  # 关系多样性
                0.2 * (geographic_diversity / 5) +  # 地理多样性
                0.2 * (business_diversity / 10) +  # 业务多样性
                0.1 * min(1.0, (controlling_parties + controlled_parties) / total_parties)  # 控制密度
            )
            
            # 计算网络中心性（简化版本）
            # 基于控制关系的强度
            centrality_score = 0
            for party in related_parties:
                relation_weight = self.relation_weights.get(party['relation_code'], 0.3)
                control_strength = max(party['control_by_related'], party['control_to_related']) / 100
                centrality_score += relation_weight * control_strength
            
            centrality_score = centrality_score / total_parties if total_parties > 0 else 0
            
            results.append({
                'stock_code': stkcd,
                'total_related_parties': total_parties,
                'parent_companies': parent_companies,
                'subsidiaries': subsidiaries,
                'sister_companies': sister_companies,
                'joint_ventures': joint_ventures,
                'individual_related': individual_related,
                'other_related': other_related,
                'controlling_parties': controlling_parties,
                'controlled_parties': controlled_parties,
                'geographic_diversity': geographic_diversity,
                'business_diversity': business_diversity,
                'network_complexity_score': complexity_score,
                'network_centrality_score': centrality_score,
                'latest_report_date': company_info['latest_report_date']
            })
        
        return results
    
    def generate_comprehensive_report(self, stock_code: str = None) -> Dict:
        """
        生成综合网络分析报告
        
        Parameters:
        -----------
        stock_code : str, optional
            特定股票代码
            
        Returns:
        --------
        Dict
            综合分析报告
        """
        indicator_1 = self.calculate_network_indicator_1(stock_code)
        indicator_2 = self.calculate_network_indicator_2(stock_code)
        
        # 合并结果
        combined_results = []
        
        # 创建indicator_2的字典便于查找
        indicator_2_dict = {item['stock_code']: item for item in indicator_2}
        
        for item1 in indicator_1:
            stkcd = item1['stock_code']
            combined_item = item1.copy()
            
            if stkcd in indicator_2_dict:
                item2 = indicator_2_dict[stkcd]
                # 添加indicator_2的字段，避免重复
                for key, value in item2.items():
                    if key not in combined_item:
                        combined_item[key] = value
            
            # 计算综合网络影响力评分
            control_score = min(1.0, combined_item['weighted_control_score'] / 100)
            complexity_score = combined_item.get('network_complexity_score', 0)
            centrality_score = combined_item.get('network_centrality_score', 0)
            
            combined_item['network_influence_score'] = (
                0.4 * control_score + 
                0.3 * complexity_score + 
                0.3 * centrality_score
            )
            
            combined_results.append(combined_item)
        
        # 计算汇总统计
        if combined_results:
            total_companies = len(combined_results)
            avg_parties = sum(r['total_related_parties'] for r in combined_results) / total_companies
            avg_control_received = sum(r['total_control_received'] for r in combined_results) / total_companies
            avg_control_given = sum(r['total_control_given'] for r in combined_results) / total_companies
            max_influence = max(r['network_influence_score'] for r in combined_results)
            
            summary_stats = {
                'total_companies_analyzed': total_companies,
                'avg_related_parties_per_company': avg_parties,
                'avg_control_received': avg_control_received,
                'avg_control_given': avg_control_given,
                'max_network_influence': max_influence
            }
        else:
            summary_stats = {}
        
        return {
            'indicator_1_results': indicator_1,
            'indicator_2_results': indicator_2,
            'combined_analysis': combined_results,
            'summary_statistics': summary_stats
        }
    
    def export_results(self, report: Dict, output_dir: str = "/workspace"):
        """
        导出分析结果到CSV文件
        """
        # 导出指标1结果
        if report['indicator_1_results']:
            indicator_1_file = f"{output_dir}/related_party_indicator_1.csv"
            self._export_to_csv(indicator_1_file, report['indicator_1_results'])
            print(f"指标1结果已导出: {indicator_1_file}")
        
        # 导出指标2结果
        if report['indicator_2_results']:
            indicator_2_file = f"{output_dir}/related_party_indicator_2.csv"
            self._export_to_csv(indicator_2_file, report['indicator_2_results'])
            print(f"指标2结果已导出: {indicator_2_file}")
        
        # 导出综合分析结果
        if report['combined_analysis']:
            combined_file = f"{output_dir}/related_party_combined_analysis.csv"
            # 简化综合结果，移除复杂的嵌套字段
            simplified_results = []
            for item in report['combined_analysis']:
                simplified_item = {k: v for k, v in item.items() 
                                if not isinstance(v, (dict, list))}
                simplified_results.append(simplified_item)
            
            self._export_to_csv(combined_file, simplified_results)
            print(f"综合分析结果已导出: {combined_file}")
    
    def _export_to_csv(self, filename: str, data: List[Dict]):
        """
        导出数据到CSV文件
        """
        if not data:
            return
            
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # 只包含非复杂类型的字段
            simple_fields = []
            for key, value in data[0].items():
                if not isinstance(value, (dict, list)):
                    simple_fields.append(key)
            
            writer = csv.DictWriter(csvfile, fieldnames=simple_fields)
            writer.writeheader()
            
            for row in data:
                simple_row = {k: v for k, v in row.items() if k in simple_fields}
                writer.writerow(simple_row)
    
    def print_analysis_summary(self, report: Dict):
        """
        打印分析摘要
        """
        print("\n" + "="*60)
        print("关联方网络分析摘要")
        print("="*60)
        
        if report['summary_statistics']:
            stats = report['summary_statistics']
            print(f"分析公司数量: {stats['total_companies_analyzed']}")
            print(f"平均关联方数量: {stats['avg_related_parties_per_company']:.1f}")
            print(f"平均接受控制权益: {stats['avg_control_received']:.2f}%")
            print(f"平均给出控制权益: {stats['avg_control_given']:.2f}%")
            print(f"最高网络影响力: {stats['max_network_influence']:.3f}")
        
        print("\n" + "-"*40)
        print("各公司网络指标详情:")
        print("-"*40)
        
        for result in report['combined_analysis']:
            print(f"\n股票代码: {result['stock_code']}")
            print(f"  关联方数量: {result['total_related_parties']}")
            print(f"  控制权益(接受): {result['total_control_received']:.2f}%")
            print(f"  控制权益(给出): {result['total_control_given']:.2f}%")
            print(f"  网络复杂度: {result.get('network_complexity_score', 0):.3f}")
            print(f"  网络影响力: {result['network_influence_score']:.3f}")

def main():
    """
    主函数 - 演示关联方网络分析
    """
    print("="*60)
    print("上市公司关联方网络分析工具")
    print("="*60)
    
    # 初始化分析器
    analyzer = RelatedPartyNetworkAnalyzer()
    
    # 加载数据（这里使用示例数据）
    print("加载关联方数据...")
    analyzer._create_sample_data()
    
    # 生成综合分析报告
    print("\n计算网络指标...")
    report = analyzer.generate_comprehensive_report()
    
    # 显示分析摘要
    analyzer.print_analysis_summary(report)
    
    # 导出结果
    print("\n导出分析结果...")
    analyzer.export_results(report)
    
    print("\n分析完成!")
    
    # 演示单个公司分析
    print("\n" + "="*40)
    print("单个公司分析示例 (000001)")
    print("="*40)
    single_company_report = analyzer.generate_comprehensive_report('000001')
    analyzer.print_analysis_summary(single_company_report)

if __name__ == "__main__":
    main()