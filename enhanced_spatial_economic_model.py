"""
Enhanced Spatial Economic Model with Demand Shock Analysis
=========================================================

This enhanced model focuses on analyzing how digital capital adoption and 
firm-to-firm networks mitigate the negative impacts of demand shocks on 
firm performance metrics including TFP, markup, markdown, and profitability.

Key Features:
- Comprehensive demand shock mechanisms
- Firm performance metrics (TFP, markup, markdown, profitability)
- Shock mitigation analysis through digital adoption and networks
- Robustness testing across different shock scenarios
- Advanced visualization of shock propagation and recovery

Author: Enhanced Version for Shock Analysis
Version: 5.0
Date: 2025
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from functools import cached_property, lru_cache
import pandas as pd
import warnings
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle
from collections import defaultdict

# Configure plotting and warnings
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')
np.random.seed(42)

# Global configuration
FIGURE_SIZE = (16, 12)
DPI = 300
SAVE_FIGURES = True
OUTPUT_DIR = Path("shock_analysis_outputs")
HIGH_DPI = 900

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class ShockConfig:
    """Configuration for demand shock scenarios."""
    name: str
    description: str
    shock_type: str  # 'uniform', 'targeted', 'cascade', 'persistent'
    shock_magnitude: float = 0.3
    shock_duration: int = 3
    recovery_rate: float = 0.1
    affected_firms: List[int] = field(default_factory=list)
    shock_persistence: float = 0.8
    network_propagation: float = 0.5
    digital_mitigation: float = 1.0
    network_mitigation: float = 1.0


@dataclass
class FirmPerformanceMetrics:
    """Container for firm performance metrics."""
    tfp: float = 0.0
    markup: float = 0.0
    markdown: float = 0.0
    profitability: float = 0.0
    market_share: float = 0.0
    productivity_growth: float = 0.0
    cost_efficiency: float = 0.0
    revenue: float = 0.0
    costs: float = 0.0
    profit_margin: float = 0.0


class DemandShockGenerator:
    """Generate various types of demand shocks."""
    
    def __init__(self, n_firms: int):
        self.n_firms = n_firms
    
    def generate_uniform_shock(self, magnitude: float, duration: int) -> np.ndarray:
        """Generate uniform demand shock affecting all firms."""
        shock_matrix = np.zeros((duration, self.n_firms))
        for t in range(duration):
            shock_matrix[t, :] = 1 - magnitude * np.exp(-0.1 * t)
        return shock_matrix
    
    def generate_targeted_shock(self, magnitude: float, duration: int, 
                              affected_firms: List[int]) -> np.ndarray:
        """Generate targeted shock affecting specific firms."""
        shock_matrix = np.ones((duration, self.n_firms))
        for t in range(duration):
            for firm in affected_firms:
                shock_matrix[t, firm] = 1 - magnitude * np.exp(-0.1 * t)
        return shock_matrix
    
    def generate_cascade_shock(self, magnitude: float, duration: int, 
                             network_weights: np.ndarray, 
                             initial_firms: List[int]) -> np.ndarray:
        """Generate cascade shock propagating through network."""
        shock_matrix = np.ones((duration, self.n_firms))
        
        # Initial shock to selected firms
        for firm in initial_firms:
            shock_matrix[0, firm] = 1 - magnitude
        
        # Cascade propagation
        for t in range(1, duration):
            for i in range(self.n_firms):
                if shock_matrix[t-1, i] < 1.0:  # If firm was shocked
                    # Propagate shock to connected firms
                    for j in range(self.n_firms):
                        if network_weights[i, j] > 0.1:  # Strong connection
                            propagation_strength = network_weights[i, j] * 0.3
                            shock_matrix[t, j] = min(shock_matrix[t, j], 
                                                   1 - magnitude * propagation_strength)
        
        return shock_matrix
    
    def generate_persistent_shock(self, magnitude: float, duration: int,
                                persistence: float) -> np.ndarray:
        """Generate persistent shock with slow recovery."""
        shock_matrix = np.zeros((duration, self.n_firms))
        for t in range(duration):
            shock_matrix[t, :] = 1 - magnitude * (persistence ** t)
        return shock_matrix


class FirmPerformanceAnalyzer:
    """Analyze firm performance metrics including TFP, markup, markdown."""
    
    def __init__(self, params):
        self.params = params
    
    def compute_tfp(self, output: np.ndarray, labor: np.ndarray, 
                   capital: np.ndarray, intermediate_inputs: np.ndarray) -> np.ndarray:
        """Compute Total Factor Productivity (TFP)."""
        # TFP = Y / (L^α * K^β * M^γ)
        tfp = output / (
            (labor ** self.params.alpha) * 
            (capital ** self.params.beta) * 
            (intermediate_inputs ** self.params.gamma)
        )
        return tfp
    
    def compute_markup(self, prices: np.ndarray, marginal_costs: np.ndarray) -> np.ndarray:
        """Compute markup as price/marginal cost."""
        markup = prices / (marginal_costs + 1e-8)
        return markup
    
    def compute_markdown(self, prices: np.ndarray, marginal_revenue: np.ndarray) -> np.ndarray:
        """Compute markdown as marginal revenue/price."""
        markdown = marginal_revenue / (prices + 1e-8)
        return markdown
    
    def compute_marginal_costs(self, wages: np.ndarray, capital_costs: np.ndarray,
                              labor: np.ndarray, capital: np.ndarray,
                              productivity: np.ndarray) -> np.ndarray:
        """Compute marginal costs for each firm."""
        # MC = w/MPL = w/(α * A * L^(α-1) * K^β * M^γ)
        marginal_product_labor = (
            self.params.alpha * productivity * 
            (labor ** (self.params.alpha - 1)) * 
            (capital ** self.params.beta)
        )
        marginal_costs = wages / (marginal_product_labor + 1e-8)
        return marginal_costs
    
    def compute_marginal_revenue(self, prices: np.ndarray, demand_elasticity: float,
                                output: np.ndarray) -> np.ndarray:
        """Compute marginal revenue for each firm."""
        # MR = P * (1 - 1/ε) where ε is demand elasticity
        marginal_revenue = prices * (1 - 1/demand_elasticity)
        return marginal_revenue
    
    def compute_profitability(self, revenue: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Compute profitability (profit/revenue)."""
        profit = revenue - costs
        profitability = profit / (revenue + 1e-8)
        return profitability
    
    def compute_market_share(self, output: np.ndarray) -> np.ndarray:
        """Compute market share for each firm."""
        total_output = np.sum(output)
        market_share = output / (total_output + 1e-8)
        return market_share
    
    def compute_cost_efficiency(self, output: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Compute cost efficiency (output/cost)."""
        cost_efficiency = output / (costs + 1e-8)
        return cost_efficiency
    
    def analyze_firm_performance(self, firm_data, spatial_data, 
                               demand_shifters: np.ndarray) -> Dict[str, np.ndarray]:
        """Comprehensive firm performance analysis."""
        
        # Compute intermediate inputs
        intermediate_inputs = np.ones(len(firm_data.output))  # Simplified
        
        # Compute TFP
        tfp = self.compute_tfp(firm_data.output, firm_data.labor, 
                              firm_data.capital, intermediate_inputs)
        
        # Compute marginal costs and revenue
        marginal_costs = self.compute_marginal_costs(
            spatial_data.wages[firm_data.locations],
            spatial_data.capital_costs[firm_data.locations],
            firm_data.labor, firm_data.capital, firm_data.productivity_base
        )
        
        marginal_revenue = self.compute_marginal_revenue(
            firm_data.prices, self.params.epsilon, firm_data.output
        )
        
        # Compute markup and markdown
        markup = self.compute_markup(firm_data.prices, marginal_costs)
        markdown = self.compute_markdown(firm_data.prices, marginal_revenue)
        
        # Compute revenue and costs
        revenue = firm_data.prices * firm_data.output
        costs = (
            spatial_data.wages[firm_data.locations] * firm_data.labor +
            spatial_data.capital_costs[firm_data.locations] * firm_data.capital +
            self.params.c_delta * (firm_data.digital_adoption ** 2)
        )
        
        # Compute other metrics
        profitability = self.compute_profitability(revenue, costs)
        market_share = self.compute_market_share(firm_data.output)
        cost_efficiency = self.compute_cost_efficiency(firm_data.output, costs)
        
        return {
            'tfp': tfp,
            'markup': markup,
            'markdown': markdown,
            'profitability': profitability,
            'market_share': market_share,
            'cost_efficiency': cost_efficiency,
            'revenue': revenue,
            'costs': costs,
            'profit_margin': (revenue - costs) / (revenue + 1e-8)
        }


class ShockMitigationAnalyzer:
    """Analyze how digital adoption and networks mitigate shock impacts."""
    
    def __init__(self):
        self.mitigation_results = {}
    
    def analyze_digital_mitigation(self, performance_baseline: Dict,
                                 performance_digital: Dict,
                                 shock_magnitude: float) -> Dict:
        """Analyze digital adoption's mitigation effects."""
        
        mitigation_effects = {}
        
        for metric in performance_baseline.keys():
            baseline_values = performance_baseline[metric]
            digital_values = performance_digital[metric]
            
            # Compute percentage change due to shock
            baseline_shock_impact = (baseline_values - np.mean(baseline_values)) / np.mean(baseline_values)
            digital_shock_impact = (digital_values - np.mean(digital_values)) / np.mean(digital_values)
            
            # Mitigation effect (reduction in negative impact)
            mitigation = (baseline_shock_impact - digital_shock_impact) / (baseline_shock_impact + 1e-8)
            mitigation_effects[metric] = {
                'baseline_impact': np.mean(baseline_shock_impact),
                'digital_impact': np.mean(digital_shock_impact),
                'mitigation_effect': np.mean(mitigation),
                'mitigation_percentage': np.mean(mitigation) * 100
            }
        
        return mitigation_effects
    
    def analyze_network_mitigation(self, performance_baseline: Dict,
                                 performance_network: Dict,
                                 shock_magnitude: float) -> Dict:
        """Analyze network effects' mitigation effects."""
        
        mitigation_effects = {}
        
        for metric in performance_baseline.keys():
            baseline_values = performance_baseline[metric]
            network_values = performance_network[metric]
            
            # Compute shock resilience
            baseline_resilience = 1 - np.abs((baseline_values - np.mean(baseline_values)) / np.mean(baseline_values))
            network_resilience = 1 - np.abs((network_values - np.mean(network_values)) / np.mean(network_values))
            
            # Network mitigation effect
            mitigation = (network_resilience - baseline_resilience) / (baseline_resilience + 1e-8)
            mitigation_effects[metric] = {
                'baseline_resilience': np.mean(baseline_resilience),
                'network_resilience': np.mean(network_resilience),
                'mitigation_effect': np.mean(mitigation),
                'mitigation_percentage': np.mean(mitigation) * 100
            }
        
        return mitigation_effects
    
    def analyze_combined_mitigation(self, performance_baseline: Dict,
                                  performance_combined: Dict,
                                  shock_magnitude: float) -> Dict:
        """Analyze combined digital and network mitigation effects."""
        
        mitigation_effects = {}
        
        for metric in performance_baseline.keys():
            baseline_values = performance_baseline[metric]
            combined_values = performance_combined[metric]
            
            # Compute volatility reduction
            baseline_volatility = np.std(baseline_values) / np.mean(baseline_values)
            combined_volatility = np.std(combined_values) / np.mean(combined_values)
            
            # Combined mitigation effect
            volatility_reduction = (baseline_volatility - combined_volatility) / (baseline_volatility + 1e-8)
            
            mitigation_effects[metric] = {
                'baseline_volatility': baseline_volatility,
                'combined_volatility': combined_volatility,
                'volatility_reduction': volatility_reduction,
                'reduction_percentage': volatility_reduction * 100
            }
        
        return mitigation_effects


class EnhancedShockAnalysisModel:
    """Enhanced model for analyzing demand shocks and mitigation effects."""
    
    def __init__(self, params, shock_config: ShockConfig):
        self.params = params
        self.shock_config = shock_config
        self.shock_generator = DemandShockGenerator(params.N)
        self.performance_analyzer = FirmPerformanceAnalyzer(params)
        self.mitigation_analyzer = ShockMitigationAnalyzer()
        
        # Initialize model components
        self._setup_model()
        
        # Store performance metrics over time
        self.performance_history = defaultdict(list)
        self.shock_history = []
    
    def _setup_model(self) -> None:
        """Initialize model components."""
        # Spatial data
        self.spatial_data = self._create_spatial_data()
        
        # Firm data
        self.firm_data = self._create_firm_data()
        
        # Setup network
        self._setup_firm_network()
        
        # Compute initial productivity
        self.firm_data.productivity_base = self._compute_agglomeration_effects()
        
        # Initialize equilibrium state
        self._initialize_state()
    
    def _create_spatial_data(self):
        """Create spatial data container."""
        distances = np.random.uniform(1.0, 3.0, (self.params.S, self.params.S))
        distances = (distances + distances.T) * 0.5
        np.fill_diagonal(distances, 0.0)
        
        wages = np.random.uniform(0.8, 1.2, self.params.S)
        capital_costs = np.random.uniform(0.9, 1.1, self.params.S)
        labor_supply = np.ones(self.params.S)
        amenities = np.ones(self.params.S)
        
        return type('SpatialData', (), {
            'distances': distances,
            'wages': wages,
            'capital_costs': capital_costs,
            'labor_supply': labor_supply,
            'amenities': amenities
        })()
    
    def _create_firm_data(self):
        """Create firm data container."""
        locations = np.random.randint(0, self.params.S, self.params.N)
        productivity_base = np.ones(self.params.N)
        network_weights = np.zeros((self.params.N, self.params.N))
        
        prices = np.random.uniform(0.8, 1.2, self.params.N)
        output = np.ones(self.params.N)
        labor = np.random.uniform(0.8, 1.2, self.params.N)
        capital = np.random.uniform(0.8, 1.2, self.params.N)
        digital_adoption = np.random.uniform(0, 0.3, self.params.N)
        profits = np.zeros(self.params.N)
        market_shares = np.zeros(self.params.N)
        
        return type('FirmData', (), {
            'locations': locations,
            'productivity_base': productivity_base,
            'network_weights': network_weights,
            'prices': prices,
            'output': output,
            'labor': labor,
            'capital': capital,
            'digital_adoption': digital_adoption,
            'profits': profits,
            'market_shares': market_shares
        })()
    
    def _setup_firm_network(self) -> None:
        """Create firm network with scenario-specific modifications."""
        network_prob = 0.3 * self.shock_config.network_mitigation
        network_mask = np.random.random((self.params.N, self.params.N)) < network_prob
        network_weights = np.random.uniform(0.1, 1.0, (self.params.N, self.params.N))
        network_weights *= network_mask
        np.fill_diagonal(network_weights, 0)
        
        # Normalize rows
        row_sums = network_weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.firm_data.network_weights = network_weights / row_sums
    
    def _compute_agglomeration_effects(self) -> np.ndarray:
        """Compute agglomeration effects with network modifications."""
        productivity = np.ones(self.params.N)
        
        for j in range(self.params.N):
            loc_j = self.firm_data.locations[j]
            spillover = 0.0
            
            for k in range(self.params.N):
                if k != j:
                    loc_k = self.firm_data.locations[k]
                    distance = self.spatial_data.distances[loc_j, loc_k]
                    network_weight = self.firm_data.network_weights[j, k]
                    spillover += network_weight * np.exp(-self.params.kappa * distance)
            
            productivity[j] = (1.0 + spillover) ** self.params.rho
        
        return productivity
    
    def _initialize_state(self) -> None:
        """Initialize state with scenario-specific modifications."""
        # Apply digital boost to initial adoption
        self.firm_data.digital_adoption *= self.shock_config.digital_mitigation
        self.firm_data.digital_adoption = np.clip(self.firm_data.digital_adoption, 0, 1)
    
    def compute_productivity(self, digital_adoption: np.ndarray) -> np.ndarray:
        """Compute productivity with digital transformation effects."""
        digital_boost = (1.0 + self.params.eta * digital_adoption) ** self.params.nu
        return self.firm_data.productivity_base * digital_boost
    
    def solve_equilibrium_with_shock(self, demand_shifters: np.ndarray, 
                                   max_iter: int = 30, tol: float = 1e-4) -> bool:
        """Solve equilibrium with demand shock effects."""
        
        for iteration in range(max_iter):
            # Store old values
            old_prices = self.firm_data.prices.copy()
            old_wages = self.spatial_data.wages.copy()
            
            # Solve single iteration
            self._solve_single_iteration(demand_shifters)
            
            # Check convergence
            price_change = np.max(np.abs(self.firm_data.prices - old_prices))
            wage_change = np.max(np.abs(self.spatial_data.wages - old_wages))
            
            if price_change < tol and wage_change < tol:
                return True
        
        return False
    
    def _solve_single_iteration(self, demand_shifters: np.ndarray) -> None:
        """Single iteration of equilibrium solving with shock effects."""
        # Compute productivity and trade costs
        productivity = self.compute_productivity(self.firm_data.digital_adoption)
        trade_costs = self._compute_trade_costs(self.firm_data.digital_adoption)
        
        # Simplified output computation with shock effects
        self.firm_data.output = (
            productivity * 
            (self.firm_data.labor ** self.params.alpha) * 
            (self.firm_data.capital ** self.params.beta) *
            demand_shifters  # Apply demand shock directly
        )
        
        # Price updates with shock consideration
        aggregate_price = np.mean(self.firm_data.prices)
        demand = demand_shifters * (self.firm_data.prices ** (-self.params.epsilon)) * self.params.Y
        
        excess_demand = (demand - self.firm_data.output) / (self.firm_data.output + 1e-8)
        self.firm_data.prices *= (1 + 0.1 * np.tanh(excess_demand))
        self.firm_data.prices = np.maximum(self.firm_data.prices, 0.1)
        
        # Wage updates
        for s in range(self.params.S):
            location_mask = self.firm_data.locations == s
            labor_demand = np.sum(self.firm_data.labor[location_mask])
            labor_supply = self.spatial_data.labor_supply[s]
            excess_labor = labor_demand - labor_supply
            self.spatial_data.wages[s] *= (1 + 0.05 * np.tanh(excess_labor))
            self.spatial_data.wages[s] = max(self.spatial_data.wages[s], 0.1)
    
    def _compute_trade_costs(self, digital_adoption: np.ndarray) -> np.ndarray:
        """Compute trade costs with digital transformation effects."""
        trade_costs = np.ones((self.params.N, self.params.N))
        
        for i in range(self.params.N):
            for j in range(self.params.N):
                if i != j:
                    loc_i, loc_j = self.firm_data.locations[i], self.firm_data.locations[j]
                    base_distance = self.spatial_data.distances[loc_i, loc_j]
                    digital_effect = min(digital_adoption[i], digital_adoption[j])
                    digital_reduction = (1.0 - self.params.theta * digital_effect) ** self.params.mu
                    trade_costs[i, j] = self.params.tau_0 * base_distance * digital_reduction
                else:
                    trade_costs[i, j] = 1e-8
        
        return trade_costs
    
    def simulate_shock_scenario(self) -> Dict:
        """Simulate complete shock scenario with performance analysis."""
        print(f"🚀 Running shock scenario: {self.shock_config.name}")
        print(f"   Shock type: {self.shock_config.shock_type}")
        print(f"   Shock magnitude: {self.shock_config.shock_magnitude}")
        
        results = {
            'performance_metrics': [],
            'shock_patterns': [],
            'mitigation_effects': {},
            'time_series': defaultdict(list)
        }
        
        # Generate shock pattern
        if self.shock_config.shock_type == 'uniform':
            shock_matrix = self.shock_generator.generate_uniform_shock(
                self.shock_config.shock_magnitude, self.shock_config.shock_duration
            )
        elif self.shock_config.shock_type == 'targeted':
            shock_matrix = self.shock_generator.generate_targeted_shock(
                self.shock_config.shock_magnitude, self.shock_config.shock_duration,
                self.shock_config.affected_firms
            )
        elif self.shock_config.shock_type == 'cascade':
            shock_matrix = self.shock_generator.generate_cascade_shock(
                self.shock_config.shock_magnitude, self.shock_config.shock_duration,
                self.firm_data.network_weights, self.shock_config.affected_firms
            )
        elif self.shock_config.shock_type == 'persistent':
            shock_matrix = self.shock_generator.generate_persistent_shock(
                self.shock_config.shock_magnitude, self.shock_config.shock_duration,
                self.shock_config.shock_persistence
            )
        
        # Store shock pattern
        self.shock_history = shock_matrix.copy()
        
        # Simulate over time periods
        for t in range(self.params.T + 1):
            # Apply shock if within shock duration
            if t < len(shock_matrix):
                demand_shifters = shock_matrix[t, :]
            else:
                # Recovery period
                recovery_factor = np.exp(-self.shock_config.recovery_rate * (t - len(shock_matrix)))
                demand_shifters = 1.0 - (1.0 - shock_matrix[-1, :]) * recovery_factor
            
            # Solve equilibrium
            converged = self.solve_equilibrium_with_shock(demand_shifters)
            
            if not converged:
                print(f"Warning: Period {t} did not converge")
            
            # Analyze firm performance
            performance_metrics = self.performance_analyzer.analyze_firm_performance(
                self.firm_data, self.spatial_data, demand_shifters
            )
            
            # Store results
            results['performance_metrics'].append(performance_metrics)
            results['shock_patterns'].append(demand_shifters.copy())
            
            # Store time series
            for metric, values in performance_metrics.items():
                results['time_series'][metric].append(values.copy())
            
            print(f"Period {t}: Avg TFP = {np.mean(performance_metrics['tfp']):.3f}, "
                  f"Avg Markup = {np.mean(performance_metrics['markup']):.3f}")
        
        return results
    
    def create_shock_analysis_visualization(self, results: Dict) -> None:
        """Create comprehensive visualization of shock analysis."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'Shock Analysis: {self.shock_config.name}', fontsize=16, fontweight='bold')
        
        # Shock pattern over time
        ax = axes[0, 0]
        shock_data = np.array(results['shock_patterns'])
        for i in range(min(5, self.params.N)):  # Show first 5 firms
            ax.plot(shock_data[:, i], label=f'Firm {i}', alpha=0.7)
        ax.set_title('Demand Shock Pattern')
        ax.set_xlabel('Time')
        ax.set_ylabel('Demand Shifter')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # TFP evolution
        ax = axes[0, 1]
        tfp_data = np.array(results['time_series']['tfp'])
        mean_tfp = np.mean(tfp_data, axis=1)
        std_tfp = np.std(tfp_data, axis=1)
        ax.plot(mean_tfp, 'b-', linewidth=2, label='Mean TFP')
        ax.fill_between(range(len(mean_tfp)), mean_tfp - std_tfp, mean_tfp + std_tfp, alpha=0.3)
        ax.set_title('TFP Evolution')
        ax.set_xlabel('Time')
        ax.set_ylabel('TFP')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Markup evolution
        ax = axes[0, 2]
        markup_data = np.array(results['time_series']['markup'])
        mean_markup = np.mean(markup_data, axis=1)
        ax.plot(mean_markup, 'r-', linewidth=2, label='Mean Markup')
        ax.set_title('Markup Evolution')
        ax.set_xlabel('Time')
        ax.set_ylabel('Markup')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Profitability evolution
        ax = axes[1, 0]
        profit_data = np.array(results['time_series']['profitability'])
        mean_profit = np.mean(profit_data, axis=1)
        ax.plot(mean_profit, 'g-', linewidth=2, label='Mean Profitability')
        ax.set_title('Profitability Evolution')
        ax.set_xlabel('Time')
        ax.set_ylabel('Profitability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Market share evolution
        ax = axes[1, 1]
        market_data = np.array(results['time_series']['market_share'])
        mean_market = np.mean(market_data, axis=1)
        ax.plot(mean_market, 'm-', linewidth=2, label='Mean Market Share')
        ax.set_title('Market Share Evolution')
        ax.set_xlabel('Time')
        ax.set_ylabel('Market Share')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cost efficiency evolution
        ax = axes[1, 2]
        cost_data = np.array(results['time_series']['cost_efficiency'])
        mean_cost = np.mean(cost_data, axis=1)
        ax.plot(mean_cost, 'c-', linewidth=2, label='Mean Cost Efficiency')
        ax.set_title('Cost Efficiency Evolution')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cost Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Performance distribution (final period)
        ax = axes[2, 0]
        final_tfp = results['time_series']['tfp'][-1]
        ax.hist(final_tfp, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Final TFP Distribution')
        ax.set_xlabel('TFP')
        ax.set_ylabel('Frequency')
        
        # Performance correlation
        ax = axes[2, 1]
        final_tfp = results['time_series']['tfp'][-1]
        final_markup = results['time_series']['markup'][-1]
        ax.scatter(final_tfp, final_markup, alpha=0.7)
        ax.set_xlabel('TFP')
        ax.set_ylabel('Markup')
        ax.set_title('TFP vs Markup (Final Period)')
        ax.grid(True, alpha=0.3)
        
        # Shock impact summary
        ax = axes[2, 2]
        shock_impact = self._compute_shock_impact_summary(results)
        metrics = list(shock_impact.keys())
        impacts = [shock_impact[metric]['impact_percentage'] for metric in metrics]
        bars = ax.bar(metrics, impacts, alpha=0.7, color='orange', edgecolor='black')
        ax.set_title('Shock Impact by Metric')
        ax.set_ylabel('Impact (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, impact in zip(bars, impacts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{impact:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        if SAVE_FIGURES:
            filename = f"shock_analysis_{self.shock_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches='tight')
            print(f"✅ Shock analysis saved to {OUTPUT_DIR / filename}")
        
        plt.show()
    
    def _compute_shock_impact_summary(self, results: Dict) -> Dict:
        """Compute summary of shock impact on different metrics."""
        impact_summary = {}
        
        for metric in results['time_series'].keys():
            data = np.array(results['time_series'][metric])
            initial_mean = np.mean(data[0, :])
            final_mean = np.mean(data[-1, :])
            impact_percentage = ((final_mean - initial_mean) / initial_mean) * 100
            
            impact_summary[metric] = {
                'initial_mean': initial_mean,
                'final_mean': final_mean,
                'impact_percentage': impact_percentage
            }
        
        return impact_summary


def run_comprehensive_shock_analysis():
    """Run comprehensive shock analysis across different scenarios."""
    print("=" * 80)
    print("🚀 Enhanced Shock Analysis Model")
    print("=" * 80)
    
    # Model parameters
    params = type('ModelParameters', (), {
        'N': 15,
        'S': 5,
        'T': 8,  # Extended time for shock analysis
        'alpha': 0.3,
        'beta': 0.3,
        'gamma': 0.4,
        'sigma': 4.0,
        'epsilon': 5.0,
        'eta': 0.2,
        'theta': 0.3,
        'mu': 0.8,
        'nu': 0.7,
        'c_delta': 0.5,
        'tau_0': 1.2,
        'rho': 0.5,
        'kappa': 0.15,
        'Y': 100.0
    })()
    
    # Define shock scenarios
    shock_scenarios = [
        ShockConfig(
            name="Baseline_Shock",
            description="Baseline scenario with uniform demand shock",
            shock_type="uniform",
            shock_magnitude=0.3,
            shock_duration=3,
            recovery_rate=0.1
        ),
        ShockConfig(
            name="Digital_Mitigation",
            description="High digital adoption mitigating shock effects",
            shock_type="uniform",
            shock_magnitude=0.3,
            shock_duration=3,
            recovery_rate=0.1,
            digital_mitigation=1.5
        ),
        ShockConfig(
            name="Network_Mitigation",
            description="Strong network effects mitigating shock propagation",
            shock_type="cascade",
            shock_magnitude=0.3,
            shock_duration=3,
            recovery_rate=0.1,
            network_mitigation=1.5,
            affected_firms=[0, 1, 2]
        ),
        ShockConfig(
            name="Combined_Mitigation",
            description="Combined digital and network mitigation",
            shock_type="cascade",
            shock_magnitude=0.3,
            shock_duration=3,
            recovery_rate=0.1,
            digital_mitigation=1.5,
            network_mitigation=1.5,
            affected_firms=[0, 1, 2]
        )
    ]
    
    # Run scenarios and collect results
    scenario_results = {}
    
    for shock_config in shock_scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {shock_config.name}")
        print(f"{'='*60}")
        
        # Create and run model
        model = EnhancedShockAnalysisModel(params, shock_config)
        results = model.simulate_shock_scenario()
        
        # Create visualization
        model.create_shock_analysis_visualization(results)
        
        # Store results
        scenario_results[shock_config.name] = {
            'config': shock_config,
            'results': results
        }
        
        print(f"✅ Scenario {shock_config.name} completed successfully")
    
    # Create mitigation analysis
    print(f"\n{'='*60}")
    print("Creating Mitigation Analysis")
    print(f"{'='*60}")
    
    create_mitigation_comparison(scenario_results)
    
    print("\n" + "=" * 80)
    print("✅ Comprehensive shock analysis complete!")
    print("📊 All shock scenarios analyzed with performance metrics")
    print("🔍 Mitigation effects quantified and visualized")
    print("=" * 80)
    
    return scenario_results


def create_mitigation_comparison(scenario_results: Dict) -> None:
    """Create comprehensive mitigation comparison analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Shock Mitigation Analysis: Digital vs Network Effects', fontsize=16, fontweight='bold')
    
    # Extract key metrics for comparison
    metrics_to_compare = ['tfp', 'markup', 'profitability', 'cost_efficiency']
    scenario_names = list(scenario_results.keys())
    
    # TFP comparison
    ax = axes[0, 0]
    for scenario_name in scenario_names:
        tfp_data = np.array(scenario_results[scenario_name]['results']['time_series']['tfp'])
        mean_tfp = np.mean(tfp_data, axis=1)
        ax.plot(mean_tfp, label=scenario_name, linewidth=2, marker='o')
    ax.set_title('TFP Evolution Across Scenarios')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean TFP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Markup comparison
    ax = axes[0, 1]
    for scenario_name in scenario_names:
        markup_data = np.array(scenario_results[scenario_name]['results']['time_series']['markup'])
        mean_markup = np.mean(markup_data, axis=1)
        ax.plot(mean_markup, label=scenario_name, linewidth=2, marker='s')
    ax.set_title('Markup Evolution Across Scenarios')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Markup')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Profitability comparison
    ax = axes[1, 0]
    for scenario_name in scenario_names:
        profit_data = np.array(scenario_results[scenario_name]['results']['time_series']['profitability'])
        mean_profit = np.mean(profit_data, axis=1)
        ax.plot(mean_profit, label=scenario_name, linewidth=2, marker='^')
    ax.set_title('Profitability Evolution Across Scenarios')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Profitability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mitigation effectiveness summary
    ax = axes[1, 1]
    mitigation_effectiveness = compute_mitigation_effectiveness(scenario_results)
    
    scenarios = list(mitigation_effectiveness.keys())
    effectiveness_scores = [mitigation_effectiveness[scenario]['overall_score'] for scenario in scenarios]
    
    bars = ax.bar(scenarios, effectiveness_scores, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.set_title('Overall Mitigation Effectiveness')
    ax.set_ylabel('Effectiveness Score')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, effectiveness_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    if SAVE_FIGURES:
        filename = f"mitigation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches='tight')
        print(f"✅ Mitigation comparison saved to {OUTPUT_DIR / filename}")
    
    plt.show()


def compute_mitigation_effectiveness(scenario_results: Dict) -> Dict:
    """Compute overall mitigation effectiveness scores."""
    effectiveness = {}
    
    baseline_results = scenario_results['Baseline_Shock']['results']
    baseline_metrics = {}
    
    # Compute baseline performance
    for metric in ['tfp', 'markup', 'profitability', 'cost_efficiency']:
        data = np.array(baseline_results['time_series'][metric])
        baseline_metrics[metric] = {
            'initial': np.mean(data[0, :]),
            'final': np.mean(data[-1, :]),
            'volatility': np.std(data) / np.mean(data)
        }
    
    # Compare other scenarios to baseline
    for scenario_name, scenario_data in scenario_results.items():
        if scenario_name == 'Baseline_Shock':
            continue
        
        results = scenario_data['results']
        scenario_effectiveness = {}
        
        for metric in ['tfp', 'markup', 'profitability', 'cost_efficiency']:
            data = np.array(results['time_series'][metric])
            
            # Compute improvements
            final_improvement = (np.mean(data[-1, :]) - baseline_metrics[metric]['final']) / baseline_metrics[metric]['final']
            volatility_reduction = (baseline_metrics[metric]['volatility'] - (np.std(data) / np.mean(data))) / baseline_metrics[metric]['volatility']
            
            scenario_effectiveness[metric] = {
                'final_improvement': final_improvement,
                'volatility_reduction': volatility_reduction,
                'combined_score': (final_improvement + volatility_reduction) / 2
            }
        
        # Overall effectiveness score
        overall_score = np.mean([scenario_effectiveness[metric]['combined_score'] for metric in scenario_effectiveness.keys()])
        effectiveness[scenario_name] = {
            'metrics': scenario_effectiveness,
            'overall_score': overall_score
        }
    
    return effectiveness


if __name__ == "__main__":
    scenario_results = run_comprehensive_shock_analysis()