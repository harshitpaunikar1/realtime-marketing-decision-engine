"""
Real-time marketing decision engine.
Computes KPIs, evaluates thresholds, and generates action recommendations.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ActionType(str, Enum):
    PAUSE_CAMPAIGN = "pause_campaign"
    INCREASE_BUDGET = "increase_budget"
    REDUCE_BUDGET = "reduce_budget"
    ESCALATE_SUPPORT = "escalate_support"
    A_B_TEST = "a_b_test"
    SEGMENT_PUSH = "segment_push"
    REVIEW_CREATIVE = "review_creative"


@dataclass
class KPISnapshot:
    campaign_id: str
    timestamp: float
    impressions: int
    clicks: int
    conversions: int
    spend: float
    revenue: float
    complaint_rate: float
    praise_rate: float
    avg_urgency: float

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0

    @property
    def cvr(self) -> float:
        return self.conversions / self.clicks if self.clicks else 0.0

    @property
    def cpc(self) -> float:
        return self.spend / self.clicks if self.clicks else 0.0

    @property
    def roas(self) -> float:
        return self.revenue / self.spend if self.spend else 0.0

    @property
    def cpa(self) -> float:
        return self.spend / self.conversions if self.conversions else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "timestamp": self.timestamp,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "conversions": self.conversions,
            "spend": round(self.spend, 2),
            "revenue": round(self.revenue, 2),
            "complaint_rate": round(self.complaint_rate, 4),
            "praise_rate": round(self.praise_rate, 4),
            "avg_urgency": round(self.avg_urgency, 4),
            "ctr": round(self.ctr, 4),
            "cvr": round(self.cvr, 4),
            "cpc": round(self.cpc, 2),
            "roas": round(self.roas, 2),
            "cpa": round(self.cpa, 2),
        }


@dataclass
class Alert:
    alert_id: str
    campaign_id: str
    severity: AlertSeverity
    message: str
    kpi: str
    kpi_value: float
    threshold: float
    created_at: float = field(default_factory=time.time)


@dataclass
class ActionRecommendation:
    campaign_id: str
    action: ActionType
    rationale: str
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThresholdConfig:
    """Configurable KPI thresholds that trigger alerts and actions."""

    DEFAULTS: Dict[str, Tuple[float, float, AlertSeverity]] = {
        "ctr": (0.005, 0.02, AlertSeverity.WARNING),
        "cvr": (0.01, 0.05, AlertSeverity.WARNING),
        "roas": (1.5, 3.0, AlertSeverity.CRITICAL),
        "complaint_rate": (0.10, 0.20, AlertSeverity.CRITICAL),
        "avg_urgency": (0.3, 0.6, AlertSeverity.WARNING),
    }

    def __init__(self, overrides: Optional[Dict] = None):
        self.thresholds = dict(self.DEFAULTS)
        if overrides:
            self.thresholds.update(overrides)

    def check(self, kpi: str, value: float) -> Optional[Tuple[AlertSeverity, float]]:
        if kpi not in self.thresholds:
            return None
        low, high, sev = self.thresholds[kpi]
        if kpi in ("complaint_rate", "avg_urgency", "cpc", "cpa"):
            if value >= high:
                return AlertSeverity.CRITICAL, high
            if value >= low:
                return sev, low
        else:
            if value <= low:
                return AlertSeverity.CRITICAL, low
            if value <= high:
                return sev, high
        return None


class ABTestEvaluator:
    """Evaluates running A/B tests using z-score significance testing."""

    def evaluate(self, control_rate: float, treatment_rate: float,
                 n_control: int, n_treatment: int,
                 alpha: float = 0.05) -> Dict[str, Any]:
        if n_control == 0 or n_treatment == 0:
            return {"significant": False, "z_score": 0.0, "lift_pct": 0.0}
        pooled = (control_rate * n_control + treatment_rate * n_treatment) / (n_control + n_treatment)
        se = np.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
        z_score = (treatment_rate - control_rate) / se if se > 0 else 0.0
        threshold_z = 1.96 if alpha == 0.05 else 2.576
        lift_pct = ((treatment_rate - control_rate) / control_rate * 100) if control_rate else 0.0
        return {
            "significant": abs(z_score) > threshold_z,
            "z_score": round(float(z_score), 3),
            "lift_pct": round(float(lift_pct), 2),
            "winner": "treatment" if z_score > threshold_z else ("control" if z_score < -threshold_z else "none"),
        }


class SegmentAnalyzer:
    """Identifies high-value and at-risk customer segments from KPI data."""

    def analyze(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if df.empty:
            return {}
        results: Dict[str, pd.DataFrame] = {}
        if "channel" in df.columns and "conversions" in df.columns:
            results["by_channel"] = (
                df.groupby("channel")
                .agg(impressions=("impressions", "sum"),
                     clicks=("clicks", "sum"),
                     conversions=("conversions", "sum"),
                     spend=("spend", "sum"),
                     revenue=("revenue", "sum"))
                .assign(ctr=lambda x: (x["clicks"] / x["impressions"].replace(0, np.nan)).round(4),
                        roas=lambda x: (x["revenue"] / x["spend"].replace(0, np.nan)).round(2))
                .reset_index()
            )
        if "campaign_id" in df.columns:
            results["by_campaign"] = (
                df.groupby("campaign_id")
                .agg(total_spend=("spend", "sum"),
                     total_revenue=("revenue", "sum"),
                     total_conversions=("conversions", "sum"))
                .assign(roas=lambda x: (x["total_revenue"] / x["total_spend"].replace(0, np.nan)).round(2))
                .reset_index()
                .sort_values("roas", ascending=False)
            )
        return results


class MarketingDecisionEngine:
    """
    Evaluates KPI snapshots, raises alerts, and recommends marketing actions.
    """

    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()
        self.ab_evaluator = ABTestEvaluator()
        self.segment_analyzer = SegmentAnalyzer()
        self._alert_history: List[Alert] = []
        self._action_history: List[ActionRecommendation] = []

    def evaluate(self, snapshot: KPISnapshot) -> Tuple[List[Alert], List[ActionRecommendation]]:
        alerts = self._check_thresholds(snapshot)
        actions = self._recommend_actions(snapshot, alerts)
        self._alert_history.extend(alerts)
        self._action_history.extend(actions)
        return alerts, actions

    def _check_thresholds(self, snap: KPISnapshot) -> List[Alert]:
        alerts = []
        kpi_values = {
            "ctr": snap.ctr,
            "cvr": snap.cvr,
            "roas": snap.roas,
            "complaint_rate": snap.complaint_rate,
            "avg_urgency": snap.avg_urgency,
        }
        for kpi, value in kpi_values.items():
            result = self.config.check(kpi, value)
            if result:
                sev, threshold = result
                alert_id = f"{snap.campaign_id}_{kpi}_{int(time.time())}"
                alerts.append(Alert(
                    alert_id=alert_id,
                    campaign_id=snap.campaign_id,
                    severity=sev,
                    message=self._alert_message(kpi, value, threshold, sev),
                    kpi=kpi,
                    kpi_value=round(value, 4),
                    threshold=threshold,
                ))
        return alerts

    def _alert_message(self, kpi: str, value: float, threshold: float,
                        sev: AlertSeverity) -> str:
        direction = "below" if value < threshold else "above"
        return (f"[{sev.upper()}] Campaign KPI '{kpi}' is {value:.4f}, "
                f"{direction} threshold {threshold:.4f}.")

    def _recommend_actions(self, snap: KPISnapshot,
                            alerts: List[Alert]) -> List[ActionRecommendation]:
        actions = []
        critical_kpis = {a.kpi for a in alerts if a.severity == AlertSeverity.CRITICAL}

        if "complaint_rate" in critical_kpis or "avg_urgency" in critical_kpis:
            actions.append(ActionRecommendation(
                campaign_id=snap.campaign_id,
                action=ActionType.ESCALATE_SUPPORT,
                rationale=f"Complaint rate {snap.complaint_rate:.1%} or urgency score elevated. "
                           "Route incoming tickets to senior support.",
                priority=1,
            ))

        if "roas" in critical_kpis and snap.roas < 1.0:
            actions.append(ActionRecommendation(
                campaign_id=snap.campaign_id,
                action=ActionType.PAUSE_CAMPAIGN,
                rationale=f"ROAS of {snap.roas:.2f} is below breakeven. "
                           "Pause to prevent further loss.",
                priority=1,
            ))
        elif "roas" in critical_kpis and snap.roas < 1.5:
            actions.append(ActionRecommendation(
                campaign_id=snap.campaign_id,
                action=ActionType.REDUCE_BUDGET,
                rationale=f"ROAS of {snap.roas:.2f} is below target {1.5}. "
                           "Reduce daily budget by 30%.",
                priority=2,
            ))

        if snap.roas > 4.0 and snap.cvr > 0.05:
            actions.append(ActionRecommendation(
                campaign_id=snap.campaign_id,
                action=ActionType.INCREASE_BUDGET,
                rationale=f"Strong ROAS {snap.roas:.2f} with CVR {snap.cvr:.1%}. "
                           "Scale budget to capture opportunity.",
                priority=2,
                metadata={"suggested_increase_pct": 25},
            ))

        if "ctr" in critical_kpis:
            actions.append(ActionRecommendation(
                campaign_id=snap.campaign_id,
                action=ActionType.REVIEW_CREATIVE,
                rationale=f"CTR of {snap.ctr:.4f} is critically low. "
                           "Creative refresh or audience targeting needed.",
                priority=2,
            ))

        return sorted(actions, key=lambda a: a.priority)

    def evaluate_ab_test(self, control: KPISnapshot,
                          treatment: KPISnapshot) -> Dict[str, Any]:
        cvr_result = self.ab_evaluator.evaluate(
            control.cvr, treatment.cvr,
            control.clicks, treatment.clicks,
        )
        ctr_result = self.ab_evaluator.evaluate(
            control.ctr, treatment.ctr,
            control.impressions, treatment.impressions,
        )
        return {
            "control_campaign": control.campaign_id,
            "treatment_campaign": treatment.campaign_id,
            "cvr_test": cvr_result,
            "ctr_test": ctr_result,
        }

    def dashboard_summary(self, snapshots: List[KPISnapshot]) -> Dict[str, Any]:
        if not snapshots:
            return {}
        df = pd.DataFrame([s.to_dict() for s in snapshots])
        return {
            "total_campaigns": len(df),
            "total_spend": round(float(df["spend"].sum()), 2),
            "total_revenue": round(float(df["revenue"].sum()), 2),
            "avg_roas": round(float(df["roas"].mean()), 2),
            "avg_ctr": round(float(df["ctr"].mean()), 4),
            "avg_cvr": round(float(df["cvr"].mean()), 4),
            "avg_complaint_rate": round(float(df["complaint_rate"].mean()), 4),
            "total_alerts": len(self._alert_history),
            "critical_alerts": sum(1 for a in self._alert_history
                                   if a.severity == AlertSeverity.CRITICAL),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)

    campaigns = ["camp_001", "camp_002", "camp_003", "camp_004"]
    snapshots = []
    for cid in campaigns:
        impressions = int(rng.integers(10000, 100000))
        clicks = int(rng.integers(200, 3000))
        conversions = int(rng.integers(5, 150))
        spend = float(rng.uniform(500, 5000))
        revenue = float(rng.uniform(300, 15000))
        snapshots.append(KPISnapshot(
            campaign_id=cid,
            timestamp=time.time(),
            impressions=impressions,
            clicks=clicks,
            conversions=conversions,
            spend=spend,
            revenue=revenue,
            complaint_rate=float(rng.uniform(0.0, 0.3)),
            praise_rate=float(rng.uniform(0.1, 0.5)),
            avg_urgency=float(rng.uniform(0.0, 0.7)),
        ))

    engine = MarketingDecisionEngine()
    print("Marketing Decision Engine - Campaign Evaluation\n")
    for snap in snapshots:
        alerts, actions = engine.evaluate(snap)
        kpis = snap.to_dict()
        print(f"Campaign {snap.campaign_id}: CTR={kpis['ctr']:.4f} CVR={kpis['cvr']:.4f} "
              f"ROAS={kpis['roas']:.2f} complaints={kpis['complaint_rate']:.1%}")
        for a in alerts:
            print(f"  ALERT [{a.severity}]: {a.message}")
        for act in actions:
            print(f"  ACTION [{act.action.value}]: {act.rationale[:80]}")

    print("\nA/B Test Evaluation:")
    ab_result = engine.evaluate_ab_test(snapshots[0], snapshots[1])
    print(json.dumps(ab_result, indent=2))

    print("\nDashboard Summary:")
    print(json.dumps(engine.dashboard_summary(snapshots), indent=2))
