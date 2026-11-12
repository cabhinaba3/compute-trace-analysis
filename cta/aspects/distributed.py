"""Distributed graph & schedule comparison aspects (Chakra-ET & GOAL)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from cta.core.bundle import resolve_bundle
from cta.core.cache import CacheStore
from cta.core.dag import solve_distributed_dag
from cta.core.models import AspectResult, CompareConfig, TrialGroup
from cta.core.registry import ADAPTERS, load_plugins, register_aspect
from cta.core.stats import compare_entity_tables, compare_scalar_metrics

_DIST_FORMATS = ("chakra_et", "chakra_host_et", "goal")


def _pick_dist_tables(tmap: dict[str, Path]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    for fmt in _DIST_FORMATS:
        if f"{fmt}.nodes" in tmap:
            return (
                pl.read_parquet(tmap[f"{fmt}.nodes"]),
                pl.read_parquet(tmap[f"{fmt}.edges"]),
                pl.read_parquet(tmap[f"{fmt}.traffic"]),
            )
    raise KeyError("No distributed graph tables found in trial bundle")


@register_aspect(
    "dag_critical_path",
    title="DAG Structure & Critical Path",
    required_formats=_DIST_FORMATS,
    any_of_formats=True,
)
class DagCriticalPathAspect:
    name = "dag_critical_path"
    title = "DAG Structure & Critical Path"
    required_formats = _DIST_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []
        scalars: dict[str, dict[str, float]] = {}

        for g in groups:
            acc: dict[str, list[float]] = {
                "critical_path_us": [],
                "cp_compute_us": [],
                "cp_exposed_comm_us": [],
                "cp_comm_mb": [],
                "dag_depth": [],
                "num_nodes": [],
                "num_edges": [],
                "mean_slack_us": [],
                "zero_slack_pct": [],
            }
            for tmap in tables[g.group_id]:
                ndf, edf, _ = _pick_dist_tables(tmap)
                st = solve_distributed_dag(ndf, edf)
                acc["critical_path_us"].append(st.critical_path_ns / 1e3)
                acc["cp_compute_us"].append(st.cp_compute_ns / 1e3)
                acc["cp_exposed_comm_us"].append(st.cp_exposed_comm_ns / 1e3)
                acc["cp_comm_mb"].append(st.cp_comm_bytes / 1e6)
                acc["dag_depth"].append(float(st.dag_depth))
                acc["num_nodes"].append(float(st.num_nodes))
                acc["num_edges"].append(float(st.num_edges))
                acc["mean_slack_us"].append(st.mean_slack_ns / 1e3)
                acc["zero_slack_pct"].append(st.zero_slack_ratio * 100.0)
            trials.append(acc)
            tot_cp = (sum(acc["cp_compute_us"]) + sum(acc["cp_exposed_comm_us"])) or 1.0
            scalars[g.label] = {
                "share_Compute": sum(acc["cp_compute_us"]) / tot_cp,
                "share_Exposed-Comm": sum(acc["cp_exposed_comm_us"]) / tot_cp,
            }

        summary = compare_scalar_metrics(
            [
                ("critical_path_us", "us", True),
                ("cp_compute_us", "us", True),
                ("cp_exposed_comm_us", "us", True),
                ("cp_comm_mb", "MB", True),
                ("dag_depth", "hops", True),
                ("num_nodes", "count", True),
                ("num_edges", "count", True),
                ("mean_slack_us", "us", False),
                ("zero_slack_pct", "%", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return AspectResult(name=self.name, title=self.title, summary=summary, scalars=scalars)


@register_aspect(
    "comm_collectives",
    title="Communication & Collective Characterization",
    required_formats=_DIST_FORMATS,
    any_of_formats=True,
)
class CommCollectivesAspect:
    name = "comm_collectives"
    title = "Communication & Collective Characterization"
    required_formats = _DIST_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []
        breakdown_dfs: list[pl.DataFrame] = []

        for g in groups:
            acc: dict[str, list[float]] = {
                "total_comm_mb": [],
                "comm_op_count": [],
                "median_msg_kb": [],
                "p95_msg_kb": [],
                "max_msg_kb": [],
            }
            per_trial_bd: list[pl.DataFrame] = []
            for tmap in tables[g.group_id]:
                ndf, _, _ = _pick_dist_tables(tmap)
                comm = ndf.filter(pl.col("op_class").is_in(["comm_coll", "comm_send"]))
                sz = comm["size_bytes"].to_numpy().astype(np.float64) if comm.height else np.zeros(0)
                acc["total_comm_mb"].append(float(sz.sum() / 1e6))
                acc["comm_op_count"].append(float(comm.height))
                acc["median_msg_kb"].append(float(np.median(sz) / 1e3) if len(sz) else 0.0)
                acc["p95_msg_kb"].append(float(np.percentile(sz, 95) / 1e3) if len(sz) else 0.0)
                acc["max_msg_kb"].append(float(np.max(sz) / 1e3) if len(sz) else 0.0)

                if comm.height:
                    bd = comm.group_by("comm_type").agg(
                        (pl.col("size_bytes").sum().cast(pl.Float64) / 1e6).alias("volume_mb"),
                        pl.len().alias("op_count"),
                    )
                    per_trial_bd.append(bd)

            trials.append(acc)
            if per_trial_bd:
                breakdown_dfs.append(
                    pl.concat(per_trial_bd)
                    .group_by("comm_type")
                    .agg(pl.col("volume_mb").mean(), pl.col("op_count").mean().round(0).cast(pl.Int64))
                )
            else:
                breakdown_dfs.append(pl.DataFrame({"comm_type": [], "volume_mb": [], "op_count": []}))

        summary = compare_scalar_metrics(
            [
                ("total_comm_mb", "MB", True),
                ("comm_op_count", "ops", True),
                ("median_msg_kb", "KB", False),
                ("p95_msg_kb", "KB", False),
                ("max_msg_kb", "KB", False),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        by_type = compare_entity_tables(labels, breakdown_dfs, key_col="comm_type", val_col="volume_mb", count_col="op_count")
        return AspectResult(name=self.name, title=self.title, summary=summary, details={"by_comm_type": by_type})


@register_aspect(
    "rank_skew",
    title="Multi-Rank Load Balance & Straggler Skew",
    required_formats=_DIST_FORMATS,
    any_of_formats=True,
)
class RankSkewAspect:
    name = "rank_skew"
    title = "Multi-Rank Load Balance & Straggler Skew"
    required_formats = _DIST_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []
        scalars: dict[str, dict[str, float]] = {}
        rank_tables: dict[str, pl.DataFrame] = {}

        for g in groups:
            acc: dict[str, list[float]] = {
                "world_size": [],
                "max_rank_compute_us": [],
                "mean_rank_compute_us": [],
                "compute_skew_ratio": [],
                "max_rank_comm_mb": [],
                "comm_jains_fairness": [],
            }
            last_per_rank = pl.DataFrame()
            for tmap in tables[g.group_id]:
                ndf, _, _ = _pick_dist_tables(tmap)
                if ndf.is_empty():
                    continue
                per_rank = (
                    ndf.group_by("rank")
                    .agg(
                        (pl.col("duration_ns").filter(pl.col("op_class") == "compute").sum().cast(pl.Float64) / 1e3).alias("compute_us"),
                        (pl.col("size_bytes").filter(pl.col("op_class").is_in(["comm_coll", "comm_send"])).sum().cast(pl.Float64) / 1e6).alias("comm_mb"),
                        pl.len().alias("op_count"),
                    )
                    .sort("rank")
                )
                last_per_rank = per_rank
                comp = per_rank["compute_us"].to_numpy()
                comm = per_rank["comm_mb"].to_numpy()
                w = len(per_rank)
                max_c = float(comp.max()) if w else 0.0
                mean_c = float(comp.mean()) if w else 0.0
                skew = (max_c / mean_c) if mean_c > 1e-12 else 1.0
                denom = float(w * np.sum(comm**2))
                jains = float((np.sum(comm) ** 2) / denom) if denom > 1e-12 else 1.0

                acc["world_size"].append(float(w))
                acc["max_rank_compute_us"].append(max_c)
                acc["mean_rank_compute_us"].append(mean_c)
                acc["compute_skew_ratio"].append(skew)
                acc["max_rank_comm_mb"].append(float(comm.max()) if w else 0.0)
                acc["comm_jains_fairness"].append(jains)

            trials.append(acc)
            rank_tables[g.label] = last_per_rank
            mean_skew = float(np.mean(acc["compute_skew_ratio"])) if acc["compute_skew_ratio"] else 1.0
            scalars[g.label] = {"share_Rank-Skew": max(0.0, (mean_skew - 1.0) / max(mean_skew, 1.0))}

        summary = compare_scalar_metrics(
            [
                ("world_size", "ranks", False),
                ("max_rank_compute_us", "us", True),
                ("mean_rank_compute_us", "us", True),
                ("compute_skew_ratio", "max/mean", True),
                ("max_rank_comm_mb", "MB", True),
                ("comm_jains_fairness", "index", False),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return AspectResult(name=self.name, title=self.title, summary=summary, details=rank_tables, scalars=scalars)


@register_aspect(
    "traffic_matrix",
    title="Rank-to-Rank Traffic Matrix & Topology Locality",
    required_formats=_DIST_FORMATS,
    any_of_formats=True,
)
class TrafficMatrixAspect:
    name = "traffic_matrix"
    title = "Rank-to-Rank Traffic Matrix & Topology Locality"
    required_formats = _DIST_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []
        pair_dfs: list[pl.DataFrame] = []

        for g in groups:
            acc: dict[str, list[float]] = {
                "active_rank_links": [],
                "hotspot_link_mb": [],
                "mean_peer_degree": [],
                "total_p2p_mb": [],
            }
            group_mats: list[pl.DataFrame] = []
            for tmap in tables[g.group_id]:
                ndf, _, tdf = _pick_dist_tables(tmap)
                w = int(ndf["rank"].n_unique()) if ndf.height else 1
                links = tdf.height
                tb_arr = tdf["total_bytes"].to_numpy() if links else np.zeros(0)
                tot_mb = float(np.sum(tb_arr) / 1e6) if links else 0.0
                hot_mb = float(np.max(tb_arr) / 1e6) if links else 0.0
                acc["active_rank_links"].append(float(links))
                acc["hotspot_link_mb"].append(hot_mb)
                acc["mean_peer_degree"].append(links / max(1, w))
                acc["total_p2p_mb"].append(tot_mb)
                if links:
                    group_mats.append(
                        tdf.with_columns(
                            pl.format("{}->{}", pl.col("src_rank"), pl.col("dst_rank")).alias("link"),
                            (pl.col("total_bytes").cast(pl.Float64) / 1e6).alias("volume_mb"),
                        ).select("link", "src_rank", "dst_rank", "volume_mb", "msg_count")
                    )

            trials.append(acc)
            if group_mats:
                pair_dfs.append(
                    pl.concat(group_mats)
                    .group_by(["link", "src_rank", "dst_rank"])
                    .agg(pl.col("volume_mb").mean(), pl.col("msg_count").mean().round(0).cast(pl.Int64))
                )
            else:
                pair_dfs.append(pl.DataFrame({"link": [], "volume_mb": [], "msg_count": []}))

        summary = compare_scalar_metrics(
            [
                ("total_p2p_mb", "MB", True),
                ("active_rank_links", "links", True),
                ("mean_peer_degree", "peers/rank", True),
                ("hotspot_link_mb", "MB", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        link_diff = compare_entity_tables(labels, pair_dfs, key_col="link", val_col="volume_mb", count_col="msg_count")
        details = {"links": link_diff, **{f"matrix_{lbl}": df for lbl, df in zip(labels, pair_dfs, strict=True)}}
        return AspectResult(name=self.name, title=self.title, summary=summary, details=details)


@dataclass(slots=True, frozen=True)
class EquivalenceVerdict:
    equivalent: bool
    world_size_chakra: int
    world_size_goal: int
    compute_rel_error: float
    comm_volume_rel_error: float
    checks: pl.DataFrame


def _expected_goal_send_bytes(ndf_chakra: pl.DataFrame, world_size: int, collective_algo: str) -> float:
    p = max(1, world_size)
    p2p_bytes = float(ndf_chakra.filter(pl.col("op_class") == "comm_send")["size_bytes"].sum())
    colls = ndf_chakra.filter(pl.col("op_class") == "comm_coll")
    if colls.is_empty() or p <= 1:
        return p2p_bytes

    algo = collective_algo.lower()
    coll_expected = 0.0
    for row in colls.select("comm_type", "size_bytes").iter_rows(named=True):
        ct = str(row["comm_type"])
        sz = float(row["size_bytes"])
        if ct == "ALL_REDUCE":
            if algo == "tree":
                # Binomial/Rabenseifner or 2x tree broadcast/reduce approx 2 * ((P-1)/P) or 2*B; standard ring/rabenseifner = 2*(P-1)/P * B
                factor = 2.0 * (p - 1) / p
            elif algo == "direct":
                factor = float(p - 1)
            else:
                factor = 2.0 * (p - 1) / p
        elif ct in ("ALL_GATHER", "REDUCE_SCATTER", "REDUCE_SCATTER_BLOCK", "ALL_TO_ALL"):
            factor = (p - 1) / p
        elif ct in ("BROADCAST", "REDUCE"):
            # Across all P ranks total bytes sent is (P-1)*B -> average ((P-1)/P)*B per rank
            factor = (p - 1) / p
        else:
            factor = 0.0
        coll_expected += sz * factor
    return p2p_bytes + coll_expected


def verify_chakra_goal_equivalence(
    chakra_source: str | Path,
    goal_source: str | Path,
    *,
    collective_algo: str = "ring",
    rel_tol: float = 0.02,
    cache_dir: str | Path = ".cta_cache",
) -> EquivalenceVerdict:
    load_plugins()
    store = CacheStore(Path(cache_dir))
    c_bundle = resolve_bundle(chakra_source)
    g_bundle = resolve_bundle(goal_source)

    c_fmt = next((f for f in ("chakra_et", "chakra_host_et") if f in c_bundle.artifacts), None)
    if c_fmt is None:
        raise ValueError(f"Expected Chakra-ET trace, found formats: {sorted(c_bundle.formats)}")
    if "goal" not in g_bundle.artifacts:
        raise ValueError(f"Expected GOAL (.goal/.bin) schedule, found formats: {sorted(g_bundle.formats)}")

    c_tables = store.get_or_ingest(c_bundle.artifacts[c_fmt], ADAPTERS[c_fmt])
    g_tables = store.get_or_ingest(g_bundle.artifacts["goal"], ADAPTERS["goal"])
    c_nodes = pl.read_parquet(c_tables["nodes"])
    g_nodes = pl.read_parquet(g_tables["nodes"])

    wc = int(c_nodes["rank"].n_unique()) if c_nodes.height else 0
    wg = int(g_nodes["rank"].n_unique()) if g_nodes.height else 0

    c_comp_ns = float(c_nodes.filter(pl.col("op_class") == "compute")["duration_ns"].sum())
    g_comp_ns = float(g_nodes.filter(pl.col("op_class") == "compute")["duration_ns"].sum())
    comp_err = abs(g_comp_ns - c_comp_ns) / max(c_comp_ns, 1.0) if (c_comp_ns > 0 or g_comp_ns > 0) else 0.0

    exp_comm_bytes = _expected_goal_send_bytes(c_nodes, max(wc, wg), collective_algo)
    act_comm_bytes = float(g_nodes.filter(pl.col("op_class") == "comm_send")["size_bytes"].sum())
    comm_err = abs(act_comm_bytes - exp_comm_bytes) / max(exp_comm_bytes, 1.0) if (exp_comm_bytes > 0 or act_comm_bytes > 0) else 0.0

    g_sends = int(g_nodes.filter(pl.col("op_class") == "comm_send").height)
    g_recvs = int(g_nodes.filter(pl.col("op_class") == "comm_recv").height)

    checks_df = pl.DataFrame(
        [
            {"check": "world_size_match", "expected": float(wc), "observed": float(wg), "rel_error": 0.0 if wc == wg else 1.0, "passed": wc == wg},
            {"check": "compute_duration_ns", "expected": c_comp_ns, "observed": g_comp_ns, "rel_error": comp_err, "passed": comp_err <= rel_tol},
            {"check": f"decomposed_comm_bytes ({collective_algo})", "expected": exp_comm_bytes, "observed": act_comm_bytes, "rel_error": comm_err, "passed": comm_err <= rel_tol},
            {"check": "goal_send_recv_balance", "expected": float(g_sends), "observed": float(g_recvs), "rel_error": 0.0 if g_sends == g_recvs else abs(g_recvs - g_sends) / max(1, g_sends), "passed": g_sends == g_recvs},
        ]
    )
    ok = bool(checks_df["passed"].all())
    return EquivalenceVerdict(
        equivalent=ok,
        world_size_chakra=wc,
        world_size_goal=wg,
        compute_rel_error=comp_err,
        comm_volume_rel_error=comm_err,
        checks=checks_df,
    )


@register_aspect(
    "schedule_equiv",
    title="Chakra-ET to GOAL Conversion Conservation",
    required_formats=("chakra_et", "goal"),
    any_of_formats=False,
)
class ScheduleEquivBundleAspect:
    name = "schedule_equiv"
    title = "Chakra-ET to GOAL Conversion Conservation"
    required_formats = ("chakra_et", "goal")
    any_of_formats = False

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []
        for g in groups:
            comp_errs, comm_errs = [], []
            for tmap in tables[g.group_id]:
                cn = pl.read_parquet(tmap["chakra_et.nodes"])
                gn = pl.read_parquet(tmap["goal.nodes"])
                w = max(int(cn["rank"].n_unique()), int(gn["rank"].n_unique()))
                cc = float(cn.filter(pl.col("op_class") == "compute")["duration_ns"].sum())
                gc = float(gn.filter(pl.col("op_class") == "compute")["duration_ns"].sum())
                comp_errs.append(abs(gc - cc) / max(cc, 1.0) * 100.0)
                eb = _expected_goal_send_bytes(cn, w, config.collective_algo)
                ab = float(gn.filter(pl.col("op_class") == "comm_send")["size_bytes"].sum())
                comm_errs.append(abs(ab - eb) / max(eb, 1.0) * 100.0)
            trials.append({"compute_mismatch_pct": comp_errs, "comm_volume_mismatch_pct": comm_errs})

        summary = compare_scalar_metrics(
            [("compute_mismatch_pct", "%", True), ("comm_volume_mismatch_pct", "%", True)],
            labels,
            trials,
            alpha=config.alpha,
        )
        return AspectResult(name=self.name, title=self.title, summary=summary)
