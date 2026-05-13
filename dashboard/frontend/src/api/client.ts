import axios from "axios";
import type {
  ClusterInfo,
  CompareRequest,
  CompareResponse,
  PolicyInfo,
  RolloutResponse,
} from "./types";

// Vite proxies /api/* to http://localhost:8000 (see vite.config.ts).
const http = axios.create({
  baseURL: "/api",
  timeout: 60_000,
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
    max_steps?: number | null;
    seed?: number;
  }): Promise<RolloutResponse> {
    const { data } = await http.post<RolloutResponse>("/rollout", req);
    return data;
  },
};
