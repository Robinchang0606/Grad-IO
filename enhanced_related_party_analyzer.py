#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版关联方网络分析工具
支持DTA文件读取和详细的网络指标计算
"""

import csv
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
import os

class EnhancedRelatedPartyAnalyzer:
    """
    增强版关联方网络分析器
    """
    
    def __init__(self):
        """
        初始化分析器
        """
        self.related_party_data = []
        self.companies = {}
        self.relation_weights = self._init_relation_weights()
        self.relation_descriptions = self._init_relation_descriptions()
        
    def _init_relation_weights(self) -> Dict[str, float]:
        """
        初始化关联关系权重
        """
        return {
            '01': 1.0,   # 上市公司的母公司
            '02': 0.9,   # 上市公司的子公司
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
    
    def _init_relation_descriptions(self) -> Dict[str, str]:
        """
        初始化关联关系描述
        """
        return {
            '01': '上市公司的母公司',
            '02': '上市公司的子公司',
            '03': '与上市公司受同一母公司控制的其他企业',
            '04': '对上市公司实施共同控制的投资方',
            '05': '对上市公司施加重大影响的投资方',
            '06': '上市公司的合营企业',
            '07': '上市公司的联营企业',
            '08': '上市公司的主要投资者个人及与其关系密切的家庭成员',
            '09': '上市公司或其母公司的关键管理人员及其关系密切的家庭成员',
            '10': '关键管理人员控制的企业',
            '11': '上市公司的关联方之间',
            '12': '其他'
        }
    
    def load_data_from_file(self, file_path: str, file_format: str = 'csv'):
        """
        从文件加载数据
        
        Parameters:
        -----------
        file_path : str
            文件路径
        file_format : str
            文件格式 ('csv', 'dta')
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            print("使用示例数据进行演示...")
            self._create_sample_data()
            return
            
        try:
            if file_format.lower() == 'dta':
                self._load_dta_file(file_path)
            else:
                self._load_csv_file(file_path)
                
            print(f"数据加载成功! 共 {len(self.related_party_data)} 条关联方记录")
            self._build_company_registry()
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            print("使用示例数据进行演示...")
            self._create_sample_data()
    
    def _load_dta_file(self, file_path: str):
        """
        加载DTA文件
        注意：这里需要安装pandas和pyreadstat库
        """
        try:
            import pandas as pd
            df = pd.read_stata(file_path)
            self.related_party_data = df.to_dict('records')
        except ImportError:
            print("警告: 需要安装pandas和pyreadstat库来读取DTA文件")
            print("请运行: pip install pandas pyreadstat")
            raise
    
    def _load_csv_file(self, file_path: str):
        """
        加载CSV文件
        """
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            self.related_party_data = list(reader)
    
    def _create_sample_data(self):
        """
        创建更丰富的示例数据
        """
        sample_data = [
            # 平安银行 (000001) 的关联方
            {
                'Stkcd': '000001', 'Repttype': '1', 'Reptdt': '2023-12-31',
                'Repart': '中国平安保险(集团)股份有限公司', 'Relation': '01',
                'Annodt': '2024-04-15', 'Relation1': '母公司',
                'Rigicy': '52.38', 'Cogicy': '0', 'Regcap': '18082192',
                'Curtype': 'CNY', 'Corebs': '保险业', 'Site': '深圳市',
                'Notes': '', 'RalatedPartyID': 'RP001'
            },
            {
                'Stkcd': '000001', 'Repttype': '1', 'Reptdt': '2023-12-31',
                'Repart': '平安科技(深圳)有限公司', 'Relation': '02',
                'Annodt': '2024-04-15', 'Relation1': '子公司',
                'Rigicy': '0', 'Cogicy': '100', 'Regcap': '500000',
                'Curtype': 'CNY', 'Corebs': '科技服务', 'Site': '深圳市',
                'Notes': '', 'RalatedPartyID': 'RP002'
            },
            {
                'Stkcd': '000001', 'Repttype': '1', 'Reptdt': '2023-12-31',
                'Repart': '平安人寿保险股份有限公司', 'Relation': '03',
                'Annodt': '2024-04-15', 'Relation1': '同一控制下企业',
                'Rigicy': '0', 'Cogicy': '0', 'Regcap': '3382000',
                'Curtype': 'CNY', 'Corebs': '人寿保险', 'Site': '深圳市',
                'Notes': '', 'RalatedPartyID': 'RP003'
            },
            {
                'Stkcd': '000001', 'Repttype': '1', 'Reptdt': '2023-12-31',
                'Repart': '平安资产管理有限责任公司', 'Relation': '03',
                'Annodt': '2024-04-15', 'Relation1': '同一控制下企业',
                'Rigicy': '0', 'Cogicy': '0', 'Regcap': '800000',
                'Curtype': 'CNY', 'Corebs': '资产管理', 'Site': '上海市',
                'Notes': '', 'RalatedPartyID': 'RP004'
            },
            {
                'Stkcd': '000001', 'Repttype': '1', 'Reptdt': '2023-12-31',
                'Repart': '平安普惠企业管理有限公司', 'Relation': '07',
                'Annodt': '2024-04-15', 'Relation1': '联营企业',
                'Rigicy': '0', 'Cogicy': '30', 'Regcap': '200000',
                'Curtype': 'CNY', 'Corebs': '企业管理咨询', 'Site': '上海市',
                'Notes': '', 'RalatedPartyID': 'RP005'
            },
            
            # 万科A (000002) 的关联方
            {
                'Stkcd': '000002', 'Repttype': '1', 'Reptdt': '2023-12-31',
                'Repart': '深圳地铁集团有限公司', 'Relation': '01',
                'Annodt': '2024-04-20', 'Relation1': '母公司',
                'Rigicy': '29.38', 'Cogicy': '0', 'Regcap': '15000000',
                'Curtype': 'CNY', 'Corebs': '城市轨道交通投资建设运营', 'Site': '深圳市',
                'Notes': '', 'RalatedPartyID': 'RP006'
            },
            {
                'Stkcd': '000002', 'Repttype': '1', 'Reptdt': '2023-12-31',
                'Repart': '万科物业发展股份有限公司', 'Relation': '02',
                'Annodt': '2024-04-20', 'Relation1': '子公司',
                'Rigicy': '0', 'Cogicy': '59.91', 'Regcap': '1200000',
                'Curtype': 'CNY', 'Corebs': '物业管理服务', 'Site': '深圳市',
                'Notes': '', 'RalatedPartyID': 'RP007'
            },
            {
                'Stkcd': '000002', 'Repttype': '1', 'Reptdt': '2023-12-31',
                'Repart': '万科企业(香港)有限公司', 'Relation': '02',
                'Annodt': '2024-04-20', 'Relation1': '子公司',
                'Rigicy': '0', 'Cogicy': '100', 'Regcap': '500000',
                'Curtype': 'HKD', 'Corebs': '投资控股', 'Site': '香港',
                'Notes': '', 'RalatedPartyID': 'RP008'
            },
            {
                'Stkcd': '000002', 'Repttype': '1', 'Reptdt': '2023-12-31',
                'Repart': '万科集团股份有限公司', 'Relation': '06',
                'Annodt': '2024-04-20', 'Relation1': '合营企业',
                'Rigicy': '0', 'Cogicy': '50', 'Regcap': '800000',
                'Curtype': 'CNY', 'Corebs': '房地产开发', 'Site': '深圳市',
                'Notes': '', 'RalatedPartyID': 'RP009'
            }
        ]
        
        self.related_party_data = sample_data
        print("增强示例数据创建成功!")
        print(f"共 {len(self.related_party_data)} 条关联方记录")
        self._build_company_registry()
    
    def _build_company_registry(self):
        """
        构建公司注册表
        """
        self.companies = {}
        
        for record in self.related_party_data:
            stkcd = record['Stkcd']
            if stkcd not in self.companies:
                self.companies[stkcd] = {
                    'stock_code': stkcd,
                    'is_listed': True,
                    'related_parties': [],
                    'latest_report_date': record['Reptdt']
                }
            
            # 处理数值字段
            def safe_float(value, default=0.0):
                try:
                    return float(value) if value and str(value).strip() else default
                except (ValueError, TypeError):
                    return default
            
            related_party = {
                'name': record['Repart'],
                'relation_code': record['Relation'],
                'relation_desc': record.get('Relation1', ''),
                'control_by_related': safe_float(record.get('Rigicy')),
                'control_to_related': safe_float(record.get('Cogicy')),
                'registered_capital': safe_float(record.get('Regcap')),
                'currency': record.get('Curtype', ''),
                'business': record.get('Corebs', ''),
                'location': record.get('Site', ''),
                'party_id': record.get('RalatedPartyID', ''),
                'report_date': record['Reptdt'],
                'announce_date': record.get('Annodt', ''),
                'notes': record.get('Notes', '')
            }
            
            self.companies[stkcd]['related_parties'].append(related_party)
        
        print(f"构建完成: {len(self.companies)} 家上市公司的关联方网络")
    
    def analyze_single_company(self, stock_code: str) -> Dict:
        """
        分析单个公司的关联方网络
        
        Parameters:
        -----------
        stock_code : str
            股票代码
            
        Returns:
        --------
        Dict
            单公司分析结果
        """
        if stock_code not in self.companies:
            return {'error': f'未找到股票代码 {stock_code} 的数据'}
        
        company_info = self.companies[stock_code]
        related_parties = company_info['related_parties']
        
        # 基础统计
        total_parties = len(related_parties)
        
        # 控制关系分析
        control_analysis = self._analyze_control_relationships(related_parties)
        
        # 网络结构分析
        network_structure = self._analyze_network_structure(related_parties)
        
        # 地理分布分析
        geographic_analysis = self._analyze_geographic_distribution(related_parties)
        
        # 业务多样性分析
        business_analysis = self._analyze_business_diversity(related_parties)
        
        # 计算综合评分
        comprehensive_score = self._calculate_comprehensive_score(
            control_analysis, network_structure, geographic_analysis, business_analysis
        )
        
        return {
            'stock_code': stock_code,
            'basic_info': {
                'total_related_parties': total_parties,
                'latest_report_date': company_info['latest_report_date']
            },
            'control_analysis': control_analysis,
            'network_structure': network_structure,
            'geographic_analysis': geographic_analysis,
            'business_analysis': business_analysis,
            'comprehensive_score': comprehensive_score,
            'detailed_parties': related_parties
        }
    
    def _analyze_control_relationships(self, parties: List[Dict]) -> Dict:
        """
        分析控制关系
        """
        total_control_received = sum(p['control_by_related'] for p in parties)
        total_control_given = sum(p['control_to_related'] for p in parties)
        
        # 按关联关系分类
        relation_breakdown = defaultdict(list)
        for party in parties:
            relation_code = party['relation_code']
            relation_breakdown[relation_code].append({
                'name': party['name'],
                'control_received': party['control_by_related'],
                'control_given': party['control_to_related'],
                'capital': party['registered_capital']
            })
        
        # 计算加权控制评分
        weighted_score = 0
        for party in parties:
            weight = self.relation_weights.get(party['relation_code'], 0.3)
            control_impact = max(party['control_by_related'], party['control_to_related'])
            weighted_score += control_impact * weight
        
        return {
            'total_control_received': total_control_received,
            'total_control_given': total_control_given,
            'net_control_position': total_control_given - total_control_received,
            'weighted_control_score': weighted_score,
            'relation_breakdown': dict(relation_breakdown),
            'control_concentration': max(total_control_received, total_control_given) / 100 if parties else 0
        }
    
    def _analyze_network_structure(self, parties: List[Dict]) -> Dict:
        """
        分析网络结构
        """
        # 按关联关系类型统计
        relation_counts = Counter(p['relation_code'] for p in parties)
        
        # 计算各类关联方数量
        structure_stats = {
            'parent_companies': relation_counts.get('01', 0),
            'subsidiaries': relation_counts.get('02', 0),
            'sister_companies': relation_counts.get('03', 0),
            'joint_control': relation_counts.get('04', 0),
            'significant_influence': relation_counts.get('05', 0),
            'joint_ventures': relation_counts.get('06', 0),
            'associates': relation_counts.get('07', 0),
            'individual_related': relation_counts.get('08', 0) + relation_counts.get('09', 0),
            'management_controlled': relation_counts.get('10', 0),
            'inter_related': relation_counts.get('11', 0),
            'others': relation_counts.get('12', 0)
        }
        
        # 计算网络复杂度
        relation_diversity = len(relation_counts)
        total_parties = len(parties)
        
        complexity_score = (
            0.3 * min(1.0, total_parties / 10) +  # 规模因子
            0.4 * min(1.0, relation_diversity / 12) +  # 关系多样性
            0.3 * min(1.0, sum(1 for p in parties if p['control_by_related'] > 0 or p['control_to_related'] > 0) / total_parties)
        ) if total_parties > 0 else 0
        
        return {
            'total_parties': total_parties,
            'relation_diversity': relation_diversity,
            'structure_breakdown': structure_stats,
            'complexity_score': complexity_score
        }
    
    def _analyze_geographic_distribution(self, parties: List[Dict]) -> Dict:
        """
        分析地理分布
        """
        locations = [p['location'] for p in parties if p['location']]
        location_counts = Counter(locations)
        
        # 识别主要地区
        domestic_locations = [loc for loc in locations if any(city in loc for city in ['市', '省', '区', '县'])]
        overseas_locations = [loc for loc in locations if loc not in domestic_locations]
        
        return {
            'total_locations': len(set(locations)),
            'domestic_count': len(domestic_locations),
            'overseas_count': len(overseas_locations),
            'location_breakdown': dict(location_counts),
            'geographic_diversity': len(set(locations)) / len(parties) if parties else 0
        }
    
    def _analyze_business_diversity(self, parties: List[Dict]) -> Dict:
        """
        分析业务多样性
        """
        businesses = [p['business'] for p in parties if p['business']]
        business_counts = Counter(businesses)
        
        # 简单的行业分类
        finance_related = len([b for b in businesses if any(word in b for word in ['银行', '保险', '投资', '基金', '证券'])])
        tech_related = len([b for b in businesses if any(word in b for word in ['科技', '技术', '软件', '信息'])])
        real_estate = len([b for b in businesses if any(word in b for word in ['房地产', '物业', '建设'])])
        manufacturing = len([b for b in businesses if any(word in b for word in ['制造', '生产', '工业'])])
        
        return {
            'total_business_types': len(set(businesses)),
            'business_breakdown': dict(business_counts),
            'industry_classification': {
                'finance_related': finance_related,
                'tech_related': tech_related,
                'real_estate': real_estate,
                'manufacturing': manufacturing,
                'others': len(businesses) - finance_related - tech_related - real_estate - manufacturing
            },
            'business_diversity': len(set(businesses)) / len(parties) if parties else 0
        }
    
    def _calculate_comprehensive_score(self, control_analysis: Dict, network_structure: Dict, 
                                     geographic_analysis: Dict, business_analysis: Dict) -> Dict:
        """
        计算综合评分
        """
        # 控制力评分 (0-1)
        control_score = min(1.0, control_analysis['weighted_control_score'] / 100)
        
        # 网络复杂度评分 (0-1)
        complexity_score = network_structure['complexity_score']
        
        # 地理多样性评分 (0-1)
        geographic_score = min(1.0, geographic_analysis['geographic_diversity'])
        
        # 业务多样性评分 (0-1)
        business_score = min(1.0, business_analysis['business_diversity'])
        
        # 综合网络影响力评分
        influence_score = (
            0.4 * control_score +
            0.3 * complexity_score +
            0.15 * geographic_score +
            0.15 * business_score
        )
        
        return {
            'control_score': control_score,
            'complexity_score': complexity_score,
            'geographic_score': geographic_score,
            'business_score': business_score,
            'overall_influence_score': influence_score,
            'risk_level': self._assess_risk_level(influence_score, control_analysis, network_structure)
        }
    
    def _assess_risk_level(self, influence_score: float, control_analysis: Dict, network_structure: Dict) -> str:
        """
        评估风险等级
        """
        high_control = control_analysis['control_concentration'] > 0.5
        complex_structure = network_structure['complexity_score'] > 0.6
        many_parties = network_structure['total_parties'] > 10
        
        if influence_score > 0.7 and (high_control or complex_structure):
            return "高风险"
        elif influence_score > 0.5 or many_parties:
            return "中风险"
        else:
            return "低风险"
    
    def generate_detailed_report(self, stock_code: str) -> str:
        """
        生成详细的分析报告
        """
        analysis = self.analyze_single_company(stock_code)
        
        if 'error' in analysis:
            return f"错误: {analysis['error']}"
        
        report = []
        report.append(f"{'='*60}")
        report.append(f"股票代码 {stock_code} 关联方网络分析报告")
        report.append(f"{'='*60}")
        
        # 基础信息
        basic = analysis['basic_info']
        report.append(f"\n【基础信息】")
        report.append(f"关联方总数: {basic['total_related_parties']}")
        report.append(f"报告日期: {basic['latest_report_date']}")
        
        # 控制关系分析
        control = analysis['control_analysis']
        report.append(f"\n【控制关系分析】")
        report.append(f"接受控制权益: {control['total_control_received']:.2f}%")
        report.append(f"对外控制权益: {control['total_control_given']:.2f}%")
        report.append(f"净控制地位: {control['net_control_position']:.2f}%")
        report.append(f"加权控制评分: {control['weighted_control_score']:.2f}")
        report.append(f"控制集中度: {control['control_concentration']:.2f}")
        
        # 网络结构分析
        network = analysis['network_structure']
        structure = network['structure_breakdown']
        report.append(f"\n【网络结构分析】")
        report.append(f"关系类型多样性: {network['relation_diversity']}")
        report.append(f"网络复杂度: {network['complexity_score']:.3f}")
        report.append(f"  - 母公司: {structure['parent_companies']}")
        report.append(f"  - 子公司: {structure['subsidiaries']}")
        report.append(f"  - 同控企业: {structure['sister_companies']}")
        report.append(f"  - 合营企业: {structure['joint_ventures']}")
        report.append(f"  - 联营企业: {structure['associates']}")
        
        # 地理分布分析
        geo = analysis['geographic_analysis']
        report.append(f"\n【地理分布分析】")
        report.append(f"地理位置数量: {geo['total_locations']}")
        report.append(f"国内关联方: {geo['domestic_count']}")
        report.append(f"海外关联方: {geo['overseas_count']}")
        report.append(f"地理多样性: {geo['geographic_diversity']:.3f}")
        
        # 业务多样性分析
        business = analysis['business_analysis']
        industry = business['industry_classification']
        report.append(f"\n【业务多样性分析】")
        report.append(f"业务类型数量: {business['total_business_types']}")
        report.append(f"业务多样性: {business['business_diversity']:.3f}")
        report.append(f"  - 金融相关: {industry['finance_related']}")
        report.append(f"  - 科技相关: {industry['tech_related']}")
        report.append(f"  - 房地产: {industry['real_estate']}")
        report.append(f"  - 制造业: {industry['manufacturing']}")
        
        # 综合评分
        score = analysis['comprehensive_score']
        report.append(f"\n【综合评分】")
        report.append(f"控制力评分: {score['control_score']:.3f}")
        report.append(f"复杂度评分: {score['complexity_score']:.3f}")
        report.append(f"地理评分: {score['geographic_score']:.3f}")
        report.append(f"业务评分: {score['business_score']:.3f}")
        report.append(f"综合影响力: {score['overall_influence_score']:.3f}")
        report.append(f"风险等级: {score['risk_level']}")
        
        return '\n'.join(report)
    
    def export_detailed_analysis(self, stock_code: str, output_dir: str = "/workspace"):
        """
        导出详细分析结果
        """
        analysis = self.analyze_single_company(stock_code)
        
        if 'error' in analysis:
            print(f"错误: {analysis['error']}")
            return
        
        # 导出详细关联方信息
        parties_file = f"{output_dir}/{stock_code}_related_parties_detail.csv"
        parties_data = []
        
        for party in analysis['detailed_parties']:
            parties_data.append({
                'stock_code': stock_code,
                'party_name': party['name'],
                'relation_code': party['relation_code'],
                'relation_desc': party['relation_desc'],
                'control_by_related': party['control_by_related'],
                'control_to_related': party['control_to_related'],
                'registered_capital': party['registered_capital'],
                'currency': party['currency'],
                'business': party['business'],
                'location': party['location'],
                'report_date': party['report_date']
            })
        
        self._export_to_csv(parties_file, parties_data)
        
        # 导出分析摘要
        summary_file = f"{output_dir}/{stock_code}_analysis_summary.json"
        summary_data = {
            'stock_code': stock_code,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'basic_info': analysis['basic_info'],
            'control_analysis': analysis['control_analysis'],
            'network_structure': analysis['network_structure'],
            'geographic_analysis': analysis['geographic_analysis'],
            'business_analysis': analysis['business_analysis'],
            'comprehensive_score': analysis['comprehensive_score']
        }
        
        # 移除不能JSON序列化的复杂对象
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if not isinstance(v, (list, dict)) or k in ['industry_classification', 'structure_breakdown']}
            return obj
        
        clean_summary = clean_for_json(summary_data)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(clean_summary, f, ensure_ascii=False, indent=2)
        
        print(f"详细分析结果已导出:")
        print(f"  - 关联方详情: {parties_file}")
        print(f"  - 分析摘要: {summary_file}")
    
    def _export_to_csv(self, filename: str, data: List[Dict]):
        """
        导出数据到CSV文件
        """
        if not data:
            return
            
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

def main():
    """
    主函数
    """
    print("="*60)
    print("增强版关联方网络分析工具")
    print("="*60)
    
    # 初始化分析器
    analyzer = EnhancedRelatedPartyAnalyzer()
    
    # 加载数据
    print("加载数据...")
    analyzer._create_sample_data()  # 使用示例数据
    
    # 分析所有公司
    print("\n分析所有公司...")
    for stock_code in analyzer.companies.keys():
        print(f"\n{'-'*40}")
        print(f"分析股票代码: {stock_code}")
        print(f"{'-'*40}")
        
        # 生成详细报告
        report = analyzer.generate_detailed_report(stock_code)
        print(report)
        
        # 导出详细分析
        analyzer.export_detailed_analysis(stock_code)
    
    print(f"\n{'='*60}")
    print("分析完成!")
    print("="*60)

if __name__ == "__main__":
    main()