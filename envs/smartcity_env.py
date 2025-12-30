import numpy as np
import pandas as pd

class SmartCityHybridEnv:
    """
    Multi-agent hybrid AC/DC microgrid (scaffold).
    Agents: A_ctrl, B_ctrl, C_ctrl (local voltage correction), EMS (battery dispatch).
    """

    def __init__(self, config):
        self.cfg = config
        self.dt = self.cfg["simulation"]["dt_seconds"]
        self.horizon = self.cfg["simulation"]["horizon_steps"]
        self.buses = self.cfg["microgrid"]["buses"]
        self.agents = self.cfg["rl"]["agents"]
        self.v_nom_ll = self.cfg["microgrid"]["v_nom_ll"]
        self.disc_actions = self.cfg["rl"]["discrete_actions"]
        self.weights = self.cfg["weights"]
        self.rng = np.random.default_rng(self.cfg["simulation"]["seed"])

        self._load_profiles()
        self.t = 0
        self.voltages_ll = np.ones((3,)) * self.v_nom_ll
        self.thd = 0.02 * np.ones((3,))
        self.soc = self.cfg["microgrid"]["battery"]["soc_init"]

    # ------------------------------------------------------------------
    # Load profiles CSV
    # ------------------------------------------------------------------
    def _load_profiles(self):
        self.profiles = pd.read_csv(self.cfg["paths"]["profiles_csv"])
        if self.cfg.get("debug", False):
            print(f"[ENV] Loaded profiles: {len(self.profiles)} rows")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self.t = 0
        self.voltages_ll = np.ones((3,)) * self.v_nom_ll
        self.thd = 0.02 * np.ones((3,))
        self.soc = self.cfg["microgrid"]["battery"]["soc_init"]
        if self.cfg.get("debug", False):
            print("[ENV] Reset called, t=0")
        return self._get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions_dict):

        # ✅ If we reached the last row, force episode termination
        if self.t >= len(self.profiles):
            if self.cfg.get("debug", False):
                print(f"[ENV] step() called at invalid t={self.t}, terminating safely.")
            dones = {agent: True for agent in self.agents}
            dones["__all__"] = True
            return self._get_obs(last_obs=True), {a:0.0 for a in self.agents}, dones, {}

        corr = np.linspace(-0.03, 0.03, self.disc_actions)
        aA = corr[int(actions_dict.get("A_ctrl", self.disc_actions//2))]
        aB = corr[int(actions_dict.get("B_ctrl", self.disc_actions//2))]
        aC = corr[int(actions_dict.get("C_ctrl", self.disc_actions//2))]

        ems_raw = int(actions_dict.get("EMS", self.disc_actions//2))
        ems_scale = np.linspace(-1, 1, self.disc_actions)[ems_raw]

        row = self.profiles.iloc[self.t]
        load_total = row["load_A_kw"] + row["load_B_kw"] + row["load_C_kw"] + row["ev_kw"]
        pv_kw = row["pv_kw"]; price = row["price_usd_per_kwh"]; co2 = row["grid_co2_kg_per_kwh"]

        # Battery model
        batt = self.cfg["microgrid"]["battery"]
        p_batt = ems_scale * batt["p_max_kw"]

        e = self.soc * batt["capacity_kwh"]
        if p_batt >= 0:        # discharge
            e -= (p_batt * self.dt / 3600) / batt["eta_dis"]
        else:                 # charge
            e -= (p_batt * self.dt / 3600) * batt["eta_ch"]
        e = np.clip(e, 0, batt["capacity_kwh"])
        self.soc = e / batt["capacity_kwh"]

        # Power balance
        net_demand = load_total - pv_kw - p_batt
        grid_import_kw = max(0.0, net_demand)

        # Voltages
        base_dev = 0.10 * (load_total / 200.0)
        devA = max(0.0, base_dev - aA - 0.02*self.rng.standard_normal())
        devB = max(0.0, base_dev - aB - 0.02*self.rng.standard_normal())
        devC = max(0.0, base_dev - aC - 0.02*self.rng.standard_normal())

        self.voltages_ll = self.v_nom_ll * (1 - np.array([devA, devB, devC]))
        self.thd = np.clip(0.01 + 0.1*base_dev - 0.5*np.array([abs(aA),abs(aB),abs(aC)]),
                           0.005, 0.15)

        # Reward terms
        dev_mean = (devA + devB + devC) / 3.0
        thd_mean = float(self.thd.mean())
        energy_kwh = grid_import_kw * self.dt / 3600
        cost = energy_kwh * price
        co2_emitted = energy_kwh * co2

        r = (- self.weights["voltage_dev"] * dev_mean
             - self.weights["thd_proxy"] * thd_mean
             - self.weights["energy_cost"] * cost
             - self.weights["co2"] * co2_emitted)

        rewards = {agent: r for agent in self.agents}

        # Time update
        self.t += 1
        done = self.t >= len(self.profiles)
        dones = {agent: done for agent in self.agents}; dones["__all__"] = done

        if done and self.cfg.get("debug", False):
            print(f"[ENV] Episode ended at t={self.t}")

        info = {
            "dev_mean": dev_mean, "thd_mean": thd_mean,
            "grid_import_kw": grid_import_kw,
            "price": price, "co2": co2, "soc": self.soc
        }
        return self._get_obs(), rewards, dones, info

    # ------------------------------------------------------------------
    # Get observation (safe even when episode ended)
    # ------------------------------------------------------------------
    def _get_obs(self, last_obs=False):
        if last_obs or self.t >= len(self.profiles):
            row = self.profiles.iloc[-1]
        else:
            row = self.profiles.iloc[self.t]

        obs_common = np.array([
            self.voltages_ll[0], self.voltages_ll[1], self.voltages_ll[2],
            self.thd[0], self.thd[1], self.thd[2],
            self.soc,
            row["price_usd_per_kwh"],
            row["grid_co2_kg_per_kwh"],
            row["pv_kw"],
            row["load_A_kw"] + row["load_B_kw"] + row["load_C_kw"] + row["ev_kw"]
        ], dtype=np.float32)

        return {agent: obs_common.copy() for agent in self.agents}
