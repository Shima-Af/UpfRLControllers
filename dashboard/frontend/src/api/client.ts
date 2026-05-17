import axios from "axios";
import type {
  ClusterInfo,
  CompareRequest,
  CompareResponse,
  DerivedThresholds,
  MultiClusterRolloutResponse,
  MultiCompareRequest,
  MultiCompareResponse,
  MultiPolicyInfo,
  MultiRolloutRequest,
  PolicyInfo,
  RolloutResponse,
} from "./types";

// Vite proxies /api/* to http://localhost:8000 (see vite.config.ts).
const http = axios.create({
  baseURL: "/api",
  timeout: 120_000,
});

export const api = {
  async listPolicies(): Promise<PolicyInfo[]> {
    const { data } = await http.get<PolicyInfo[]>("/policies");
    return data;
  },
  async listClusters(): Promise<ClusterInfo[]> {
    const { data } = await http.get<ClusterInfo[]>("/clusters");
    return data;
  },
  async compare(req: CompareRequest): Promise<CompareResponse> {
    const { data } = await http.post<CompareResponse>("/compare", req);
    return data;
  },
  async rollout(req: {
    cluster_idx: number;
    horizon_idx: number;
    policy: string;
    threshold_gbps?: number;
    hysteresis_band_mbps?: number;
    hysteresis_cooldown_steps?: number;
    max_steps?: number | null;
    seed?: number;
    split?: string;
  }): Promise<RolloutResponse> {
    const { data } = await http.post<RolloutResponse>("/rollout", req);
    return data;
  },

  // ----- Multi-cluster (Phase 7) -----
  async listMultiPolicies(): Promise<MultiPolicyInfo[]> {
    const { data } = await http.get<MultiPolicyInfo[]>("/multi/policies");
    return data;
  },
  async derivedThresholds(): Promise<DerivedThresholds> {
    const { data } = await http.get<DerivedThresholds>("/multi/derived");
    return data;
  },
  async multiRollout(
    req: MultiRolloutRequest,
  ): Promise<MultiClusterRolloutResponse> {
    const { data } = await http.post<MultiClusterRolloutResponse>(
      "/multi/rollout",
      req,
    );
    return data;
  },
  async multiCompare(req: MultiCompareRequest): Promise<MultiCompareResponse> {
    const { data } = await http.post<MultiCompareResponse>(
      "/multi/compare",
      req,
    );
    return data;
  },
};
