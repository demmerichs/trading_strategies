#!/usr/bin/env python3

from typing import Any
import numpy as np

TICKS_PER_YEAR = 10000

class Market:
    def __init__(self, agents, cash_injection, market_sim, nbr_simulations=1) -> None:
        self.agents = agents
        self.cash_injection = cash_injection
        self.market_sim = market_sim
        self.tick_count = 0
        self.market_value = np.ones(nbr_simulations, dtype=np.float32)
        for agent in self.agents:
            agent.cash = np.zeros(nbr_simulations, dtype=np.float32)
            agent.depot = np.zeros(nbr_simulations, dtype=np.float32)

    def trade(self, agent):
        order = agent.order(self.market_value, self.tick_count)
        order_value = self.market_value * order
        order_value = np.minimum(order_value, agent.cash)
        order = order_value / self.market_value
        agent.cash -= order_value
        agent.depot += order

    def tick(self):
        cash_injection = self.cash_injection(self.tick_count)
        for agent in self.agents:
            agent.cash += cash_injection
            self.trade(agent)
        self.market_sim(self.market_value, self.tick_count)
        self.tick_count += 1

class SimpleCashInjection:
    def __init__(self):
        self.cash_per_year = 10000.0
        self.payouts_per_year = 500
        assert TICKS_PER_YEAR % self.payouts_per_year == 0
        self.payout_every_n_ticks = TICKS_PER_YEAR // self.payouts_per_year
        self.payout = self.cash_per_year / self.payouts_per_year

    def __call__(self, tick) -> float:
        if tick % self.payout_every_n_ticks == 0:
            return self.payout
        return 0.0

    def __str__(self):
        ans = "cash per year\t%s\n" % self.cash_per_year
        ans += "nbr payouts per year\t%s\n" % self.payouts_per_year
        ans += "regular payout\t%s\n" % self.payout
        return ans

class RealSimpleMarketSim:
    def __init__(self):
        self.annual_growth = 0.11
        self.annual_volatility = 0.15
        # ag = (1+tg)^TPY - 1
        # (ag + 1)^(1/TPY) - 1 = tg
        self.tick_growth = (self.annual_growth + 1.0)**(1.0/TICKS_PER_YEAR) - 1.0
        self.tick_volatility = self.annual_volatility / np.sqrt(TICKS_PER_YEAR)

    def __call__(self, market_value, _tick):
        sim_shape = market_value.shape
        returns = np.random.normal(loc=self.tick_growth, scale=self.tick_volatility, size=sim_shape)
        market_value *= (1.0 + returns)

class Agent:
    def name(self):
        return self.__class__.__name__

    def get_total_value(self, market_value):
        return self.cash + self.depot * market_value

    def get_portfolio(self, market_value):
        portfolio = dict(
            cash=self.cash.mean(),
            depot=self.depot.mean(),
            total_value=self.get_total_value(market_value).mean(),
        )
        return portfolio

class KeepCashAgent(Agent):
    def order(self, _market_value, _tick):
        return np.zeros_like(self.cash)

class AllInAgent(Agent):
    def order(self, market_value, _tick):
        return self.cash / market_value

class LimitOrderAgent(Agent):
    def __init__(self) -> None:
        self.limit_wait_ticks = 10
        self.limit_ratio_value = 0.9999
    
    def order(self, market_value, tick):
        if tick % self.limit_wait_ticks == 0:
            self.limit = market_value * self.limit_ratio_value
        order = self.cash / market_value
        order *= (market_value <= self.limit)
        return order


if __name__ == '__main__':
    market = Market(
        agents=[KeepCashAgent(), AllInAgent(), LimitOrderAgent()],
        cash_injection=SimpleCashInjection(),
        market_sim=RealSimpleMarketSim(),
        nbr_simulations=1000,
    )

    print(market.cash_injection)
    sim_years = 20
    for year in range(1, 1+sim_years):
        for tick in range(TICKS_PER_YEAR):
            market.tick()
        print()
        print("year", year, "\tmarket value:", market.market_value.mean())
        for agent in market.agents:
            print(agent.__class__.__name__, agent.get_portfolio(market.market_value))
        for i, agenta in enumerate(market.agents):
            for j, agentb in enumerate(market.agents):
                if i >= j: continue
                awins = agenta.get_total_value(market.market_value) > agentb.get_total_value(market.market_value)
                bwins = agenta.get_total_value(market.market_value) < agentb.get_total_value(market.market_value)
                print(agenta.name(), "vs", agentb.name(), ":", awins.sum(), "-", bwins.sum(), "or relative", "%.2f-%.2f" % (awins.mean(), bwins.mean()))
