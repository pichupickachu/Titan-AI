#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <string>
#include <algorithm>
#include <iostream>
#include <map>

namespace py = pybind11;

// =========================================================
// 1. DATA STRUCTURES & TYPES
// =========================================================

struct Wall {
    double price;
    double vol;
};

struct WallsResult {
    std::vector<Wall> support;
    std::vector<Wall> resistance;
};

struct MarketInstruction {
    std::string regime;
    std::vector<std::string> strategy_allowed;
    double min_score;
    int ai_confidence;
};

// 🔥 UPDATED SIGNAL STRUCTURE
struct SignalResult {
    double score;
    std::string strategy;
    std::vector<std::string> reasons;
    double rsi;
    double z_score_rsi; // Statistical deviation of RSI
    double z_score_vol; // Statistical deviation of Volume
    bool is_green;

    // Diagnostic fields
    double atr;          // Volatility
    double trend_slope;  // Trend Angle (Linear Regression Slope)
};

struct LiquidationLevel {
    double price;
    std::string type;
    std::string leverage;
};

struct SmartLevel {
    double price;
    int strength;
};

struct BBResult { double upper; double middle; double lower; };

struct MicroStructureSignal {
    double imbalance; // Order Book Imbalance (-1 to 1)
    double spread;    // Bid-Ask Spread
    double wmid;      // Weighted Mid Price
};

struct TapeAnalysis {
    double buy_vol;
    double sell_vol;
    double delta;
    int whale_buys;
    double pressure;
};

// =========================================================
// 2. MATHEMATICAL LIBRARY (MATH LIB)
// =========================================================

// --- Z-SCORE ---
double calculate_z_score(const std::vector<double>& data, int period) {
    if (data.size() < period) return 0.0;
    double sum = 0.0;
    int start = data.size() - period;
    for(int i = start; i < data.size(); ++i) sum += data[i];
    double mean = sum / period;
    double sq_sum = 0.0;
    for(int i = start; i < data.size(); ++i) sq_sum += std::pow(data[i] - mean, 2);
    double std_dev = std::sqrt(sq_sum / period);
    if (std_dev == 0) return 0.0;
    return (data.back() - mean) / std_dev;
}

// --- EMA ---
double calculate_last_ema(const std::vector<double>& prices, int period) {
    if (prices.size() < period) return 0.0;
    double k = 2.0 / (period + 1);
    double ema = std::accumulate(prices.begin(), prices.begin() + period, 0.0) / period;
    for(size_t i=period; i<prices.size(); ++i) ema = (prices[i] - ema) * k + ema;
    return ema;
}

// --- RSI ---
std::vector<double> calculate_rsi_vector(const std::vector<double>& prices, int period = 14) {
    std::vector<double> rsi(prices.size(), 50.0);
    if (prices.size() <= period) return rsi;
    double gain = 0.0, loss = 0.0;
    for (int i = 1; i <= period; ++i) {
        double diff = prices[i] - prices[i - 1];
        if (diff > 0) gain += diff; else loss -= diff;
    }
    double avg_gain = gain / period;
    double avg_loss = loss / period;
    for (size_t i = period + 1; i < prices.size(); ++i) {
        double diff = prices[i] - prices[i - 1];
        double g = (diff > 0) ? diff : 0.0;
        double l = (diff < 0) ? -diff : 0.0;
        avg_gain = (avg_gain * (period - 1) + g) / period;
        avg_loss = (avg_loss * (period - 1) + l) / period;
        double rs = (avg_loss == 0) ? 100.0 : avg_gain / avg_loss;
        rsi[i] = 100.0 - (100.0 / (1.0 + rs));
    }
    return rsi;
}

// --- BOLLINGER BANDS ---
BBResult get_last_bb(const std::vector<double>& prices, int period = 20, double mult = 2.0) {
    if (prices.size() < period) return {0,0,0};
    double sum = 0.0;
    for(size_t i = prices.size() - period; i < prices.size(); ++i) sum += prices[i];
    double mean = sum / period;
    double sq_sum = 0.0;
    for(size_t i = prices.size() - period; i < prices.size(); ++i) sq_sum += std::pow(prices[i] - mean, 2);
    double std = std::sqrt(sq_sum / period);
    return { mean + std * mult, mean, mean - std * mult };
}

// --- 🔥 NEW: ATR (VOLATILITY) ---
double calculate_last_atr(const std::vector<double>& highs, const std::vector<double>& lows, const std::vector<double>& closes, int period = 14) {
    if (closes.size() < period + 1) return 0.0;
    std::vector<double> tr;
    for(size_t i = 1; i < closes.size(); ++i) {
        double hl = highs[i] - lows[i];
        double hc = std::abs(highs[i] - closes[i-1]);
        double lc = std::abs(lows[i] - closes[i-1]);
        tr.push_back(std::max({hl, hc, lc}));
    }
    double atr = 0.0;
    int start = tr.size() - period;
    if (start < 0) start = 0;
    for(int i=start; i<tr.size(); ++i) atr += tr[i];
    return atr / period;
}

// --- 🔥 NEW: LINEAR REGRESSION SLOPE (TREND ANGLE) ---
double calculate_slope(const std::vector<double>& prices, int period = 10) {
    if (prices.size() < period) return 0.0;
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int n = period;
    int start = prices.size() - n;

    for (int i = 0; i < n; ++i) {
        double x = i;
        double y = prices[start + i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    // Return normalized slope (per mille)
    return (slope / prices.back()) * 1000.0;
}

// --- ADX ---
double calculate_last_adx(const std::vector<double>& highs, const std::vector<double>& lows, const std::vector<double>& closes, int period=14) {
    if (closes.size() < period * 2) return 0.0;
    std::vector<double> tr(closes.size(), 0.0);
    std::vector<double> plus_dm(closes.size(), 0.0);
    std::vector<double> minus_dm(closes.size(), 0.0);
    for (size_t i = 1; i < closes.size(); ++i) {
        double hl = highs[i] - lows[i];
        double hc = std::abs(highs[i] - closes[i-1]);
        double lc = std::abs(lows[i] - closes[i-1]);
        tr[i] = std::max({hl, hc, lc});
        double um = highs[i] - highs[i-1];
        double dm = lows[i-1] - lows[i];
        if (um > dm && um > 0) plus_dm[i] = um;
        if (dm > um && dm > 0) minus_dm[i] = dm;
    }

    double tr_s = 0, p_s = 0, m_s = 0;
    for(int i=1; i<=period; ++i) { tr_s+=tr[i]; p_s+=plus_dm[i]; m_s+=minus_dm[i]; }

    std::vector<double> dx(closes.size(), 0.0);
    for(size_t i=period+1; i<closes.size(); ++i) {
        tr_s = tr_s - (tr_s/period) + tr[i];
        p_s = p_s - (p_s/period) + plus_dm[i];
        m_s = m_s - (m_s/period) + minus_dm[i];
        if (tr_s == 0) continue;
        double pdi = 100*p_s/tr_s;
        double mdi = 100*m_s/tr_s;
        if (pdi+mdi == 0) continue;
        dx[i] = 100 * std::abs(pdi-mdi)/(pdi+mdi);
    }
    double adx = 0;
    for(size_t i=period; i<period*2; ++i) adx += dx[i];
    adx /= period;
    for(size_t i=period*2; i<closes.size(); ++i) adx = (adx*(period-1)+dx[i])/period;
    return adx;
}

// =========================================================
// 3. CORE LOGIC
// =========================================================

// --- SIGNAL GENERATOR (TERMINATOR VERSION) ---
SignalResult get_trading_signal(
    const std::vector<double>& opens, const std::vector<double>& highs,
    const std::vector<double>& lows, const std::vector<double>& closes,
    const std::vector<double>& volumes, std::string btc_regime,
    double btc_change_pct
) {
    SignalResult res;
    // 🔥 BASE SCORE 50 (TO PREVENT BOT SILENCE)
    res.score = 50;
    res.strategy = "WAIT";
    res.is_green = false;
    res.z_score_rsi = 0; res.z_score_vol = 0; res.rsi = 50;
    res.atr = 0; res.trend_slope = 0;

    if (closes.size() < 60) {
        res.score = 0;
        res.reasons.push_back("NOT_ENOUGH_DATA_60");
        return res;
    }

    double last_close = closes.back();
    double last_open = opens.back();
    res.is_green = (last_close > last_open);
    double coin_change_pct = (last_close - last_open) / last_open;

    // Calculate indicators
    std::vector<double> rsi_vec = calculate_rsi_vector(closes, 14);
    res.rsi = rsi_vec.back();
    BBResult bb = get_last_bb(closes, 20, 2.0);

    // New metrics
    res.atr = calculate_last_atr(highs, lows, closes, 14);
    res.trend_slope = calculate_slope(closes, 10);
    double adx = calculate_last_adx(highs, lows, closes, 14);

    res.z_score_rsi = calculate_z_score(rsi_vec, 50);
    res.z_score_vol = calculate_z_score(volumes, 50);

    // Base score correction
    bool trend_up = (last_close > bb.middle);
    if (trend_up) res.score += 5; else res.score -= 5;

    // Consider Trend Slope
    if (res.trend_slope > 1.0) res.score += 5;
    if (res.trend_slope < -1.0) res.score -= 5;

    // Alpha (Strength relative to BTC)
    double alpha = coin_change_pct - btc_change_pct;

    // ========================================
    // ⚔️ STRATEGIES
    // ========================================

    // 1. SNIPE (Enhanced Version)
    if (trend_up && res.trend_slope > 0.0) {
        if (res.z_score_rsi > 0.0 && res.z_score_rsi < 2.5) {
            // If market is calm (low ATR), require less volume
            double vol_threshold = (alpha > 0.01) ? 1.0 : 1.5;

            if (res.z_score_vol > vol_threshold && res.is_green) {
                res.strategy = "SNIPE";
                res.score = 75;
                res.reasons.push_back("MOMENTUM_SPIKE");

                if (alpha > 0.015) {
                    res.score += 10;
                    res.reasons.push_back("ALPHA_LEADER");
                }
                if (res.trend_slope > 2.0) {
                    res.score += 5;
                    res.reasons.push_back("AGGRESSIVE_SLOPE");
                }
            }
        }
    }

    // 2. SAFE_REVERSAL (Bottom Fishing)
    if (res.z_score_rsi < -1.5 && last_close < bb.lower * 1.01) {
        // Protection against catching falling knives (slope shouldn't be vertical)
        if (res.trend_slope > -6.0) {
            res.strategy = "SAFE_REVERSAL";
            res.score = 70;
            res.reasons.push_back("STATISTICAL_OVERSOLD");

            if (res.is_green && res.z_score_vol > 0.5) {
                res.score += 10;
                res.reasons.push_back("VOL_CONFIRMATION");
            }
        }
    }

    // 3. CHANNEL_BOUNCE (Ranging Market Strategy)
    // Works if ADX is weak or BTC regime is FLAT
    bool is_flat_market = (adx < 25 || btc_regime == "FLAT");
    if (is_flat_market) {
        double bb_width = (bb.upper - bb.lower) / bb.middle;

        // Channel must be wide enough (> 1.2%) to be profitable
        if (bb_width > 0.012) {
            double pos_in_bb = (last_close - bb.lower) / (bb.upper - bb.lower);

            // If at channel bottom (< 20%) and RSI not overheated
            if (pos_in_bb < 0.20 && res.rsi < 45) {
                if (res.is_green) {
                    res.strategy = "CHANNEL_BOUNCE";
                    res.score = 65;
                    res.reasons.push_back("FLAT_BOTTOM_BUY");
                }
            }
        }
    }

    // 4. SCALPING (Backup Option)
    if (res.strategy == "WAIT" && res.rsi > 42 && res.rsi < 58) {
         if (res.z_score_vol > 1.2 && alpha > 0.004) {
             res.strategy = "SCALPING";
             res.score = 60;
             res.reasons.push_back("MICRO_ALPHA");
         }
    }

    // ☠️ DANGER FILTERS
    if (res.z_score_rsi > 3.0) { res.score = 0; res.reasons.push_back("DANGER_RSI_EXTREME"); }
    if (res.trend_slope < -8.0) { res.score = 0; res.reasons.push_back("DANGER_CRASH"); }

    return res;
}

// --- MARKET REGIME DETECTION ---
MarketInstruction get_market_instruction_cpp(
    const std::vector<double>& highs, const std::vector<double>& lows, const std::vector<double>& closes
) {
    MarketInstruction res;
    res.regime = "ADAPTIVE";
    res.strategy_allowed = {"SNIPE", "SAFE_REVERSAL", "SCALPING", "CHANNEL_BOUNCE"};
    res.min_score = 60;
    res.ai_confidence = 7;

    if (closes.size() < 50) return res;

    double adx = calculate_last_adx(highs, lows, closes, 14);
    double ema50 = calculate_last_ema(closes, 50);

    if (adx > 35 && closes.back() < ema50) {
        res.regime = "CRASH";
        res.strategy_allowed = {};
        res.min_score = 999;
    } else if (adx < 25) {
        res.regime = "FLAT";
    }

    return res;
}

// =========================================================
// 4. HELPER FUNCTIONS (FULL VERSIONS)
// =========================================================

WallsResult detect_liquidity_walls_cpp(const std::vector<std::vector<double>>& bids, const std::vector<std::vector<double>>& asks, double current_price) {
    WallsResult res;
    if (bids.empty() || asks.empty()) return res;
    double total_bid = 0, total_ask = 0;
    for(const auto& b : bids) total_bid += b[1];
    for(const auto& a : asks) total_ask += a[1];
    double avg_bid = total_bid / bids.size();
    double avg_ask = total_ask / asks.size();
    double thr_bid = avg_bid * 5.0;
    double thr_ask = avg_ask * 5.0;

    for(const auto& b : bids) {
        if(b[1] > thr_bid && (current_price - b[0])/current_price < 0.05) res.support.push_back({b[0], b[1]});
    }
    for(const auto& a : asks) {
        if(a[1] > thr_ask && (a[0] - current_price)/current_price < 0.05) res.resistance.push_back({a[0], a[1]});
    }
    // Sort by volume
    std::sort(res.support.begin(), res.support.end(), [](const Wall& a, const Wall& b){ return a.vol > b.vol; });
    std::sort(res.resistance.begin(), res.resistance.end(), [](const Wall& a, const Wall& b){ return a.vol > b.vol; });
    if(res.support.size() > 3) res.support.resize(3);
    if(res.resistance.size() > 3) res.resistance.resize(3);
    return res;
}

std::vector<LiquidationLevel> get_liquidation_heatmap(const std::vector<double>& highs, const std::vector<double>& lows, int window = 50) {
    std::vector<LiquidationLevel> levels;
    if (highs.size() < window) return levels;
    double max_h = -1.0, min_l = 100000000.0;
    int start = highs.size() - window;
    for(int i=start; i<highs.size(); ++i) {
        if(highs[i]>max_h) max_h=highs[i];
        if(lows[i]<min_l) min_l=lows[i];
    }
    // Standard leverages
    struct L {int l; double m; std::string n;};
    std::vector<L> levs = {{100,0.01,"100x"}, {50,0.02,"50x"}, {25,0.04,"25x"}};

    for(const auto& x : levs) levels.push_back({max_h*(1.0+x.m), "SHORT_LIQ", x.n});
    for(const auto& x : levs) levels.push_back({min_l*(1.0-x.m), "LONG_LIQ", x.n});
    std::sort(levels.begin(), levels.end(), [](const LiquidationLevel& a, const LiquidationLevel& b){ return a.price < b.price; });
    return levels;
}

// 🔥 FULL SMART MONEY VERSION (BUCKETS)
std::vector<SmartLevel> find_smart_money_levels(const std::vector<double>& prices, double tolerance_pct = 0.005) {
    std::vector<SmartLevel> levels;
    if (prices.empty()) return levels;
    double min_p = prices[0], max_p = prices[0];
    for (double p : prices) { if(p<min_p) min_p=p; if(p>max_p) max_p=p; }

    int buckets = 100;
    double step = (max_p - min_p) / buckets;
    if (step == 0) return levels;

    std::vector<int> counts(buckets, 0);
    for (double p : prices) {
        int idx = (int)((p - min_p) / step);
        if (idx >= buckets) idx = buckets - 1;
        counts[idx]++;
    }

    for (int i = 1; i < buckets - 1; ++i) {
        if (counts[i] > counts[i-1] && counts[i] > counts[i+1]) {
            if (counts[i] > prices.size() * 0.02) {
                levels.push_back({min_p + (i * step) + (step / 2.0), counts[i]});
            }
        }
    }
    std::sort(levels.begin(), levels.end(), [](const SmartLevel& a, const SmartLevel& b){ return a.strength > b.strength; });
    if (levels.size() > 5) levels.resize(5);
    return levels;
}

MicroStructureSignal analyze_order_book_microstructure(
    const std::vector<std::vector<double>>& bids,
    const std::vector<std::vector<double>>& asks
) {
    MicroStructureSignal res = {0.0, 0.0, 0.0};
    if (bids.empty() || asks.empty()) return res;

    double best_bid = bids[0][0];
    double best_ask = asks[0][0];
    res.spread = best_ask - best_bid;

    double q_bid = bids[0][1];
    double q_ask = asks[0][1];
    res.wmid = (best_bid * q_ask + best_ask * q_bid) / (q_bid + q_ask);

    double vol_bids = 0;
    double vol_asks = 0;
    int depth = std::min((int)bids.size(), (int)asks.size());
    if (depth > 10) depth = 10;

    for(int i=0; i<depth; ++i) {
        double weight = 1.0 / (i + 1); // Weight decays with depth
        vol_bids += bids[i][1] * weight;
        vol_asks += asks[i][1] * weight;
    }
    res.imbalance = (vol_bids - vol_asks) / (vol_bids + vol_asks);
    return res;
}

TapeAnalysis analyze_tape_momentum(const std::vector<std::vector<double>>& trades) {
    TapeAnalysis res = {0, 0, 0, 0, 0};
    if (trades.empty()) return res;
    double threshold = 5000.0;

    for (const auto& t : trades) {
        double price = t[0];
        double amount = t[1];
        double side = t[2];
        double usd_val = price * amount;

        if (side > 0) {
            res.buy_vol += usd_val;
            if (usd_val > threshold) res.whale_buys++;
        } else {
            res.sell_vol += usd_val;
        }
    }
    res.delta = res.buy_vol - res.sell_vol;
    double total_vol = res.buy_vol + res.sell_vol;
    if (total_vol > 0) {
        double buy_ratio = res.buy_vol / total_vol;
        res.pressure = buy_ratio * 100.0;
        res.pressure += res.whale_buys * 5.0;
    }
    return res;
}

// =========================================================
// 5. PYTHON BINDING (INTERFACE)
// =========================================================
PYBIND11_MODULE(titan_math, m) {
    m.doc() = "Titanium Core v4.0 (Full Power)";

    // Data Structures
    py::class_<SignalResult>(m, "SignalResult")
        .def_readonly("score", &SignalResult::score)
        .def_readonly("strategy", &SignalResult::strategy)
        .def_readonly("reasons", &SignalResult::reasons)
        .def_readonly("rsi", &SignalResult::rsi)
        .def_readonly("z_score_rsi", &SignalResult::z_score_rsi)
        .def_readonly("z_score_vol", &SignalResult::z_score_vol)
        .def_readonly("is_green", &SignalResult::is_green)
        .def_readonly("atr", &SignalResult::atr)
        .def_readonly("trend_slope", &SignalResult::trend_slope);

    py::class_<MarketInstruction>(m, "MarketInstruction")
        .def_readonly("regime", &MarketInstruction::regime)
        .def_readonly("strategy_allowed", &MarketInstruction::strategy_allowed)
        .def_readonly("min_score", &MarketInstruction::min_score)
        .def_readonly("ai_confidence", &MarketInstruction::ai_confidence);

    py::class_<Wall>(m, "Wall")
        .def_readonly("price", &Wall::price)
        .def_readonly("vol", &Wall::vol);

    py::class_<WallsResult>(m, "WallsResult")
        .def_readonly("support", &WallsResult::support)
        .def_readonly("resistance", &WallsResult::resistance);

    py::class_<LiquidationLevel>(m, "LiquidationLevel")
        .def_readonly("price", &LiquidationLevel::price)
        .def_readonly("type", &LiquidationLevel::type)
        .def_readonly("leverage", &LiquidationLevel::leverage);

    py::class_<SmartLevel>(m, "SmartLevel")
        .def_readonly("price", &SmartLevel::price)
        .def_readonly("strength", &SmartLevel::strength);

    py::class_<MicroStructureSignal>(m, "MicroStructureSignal")
        .def_readonly("imbalance", &MicroStructureSignal::imbalance)
        .def_readonly("spread", &MicroStructureSignal::spread)
        .def_readonly("wmid", &MicroStructureSignal::wmid);

    py::class_<TapeAnalysis>(m, "TapeAnalysis")
        .def_readonly("delta", &TapeAnalysis::delta)
        .def_readonly("whale_buys", &TapeAnalysis::whale_buys)
        .def_readonly("pressure", &TapeAnalysis::pressure);

    // Functions
    m.def("get_trading_signal", &get_trading_signal);
    m.def("get_market_instruction", &get_market_instruction_cpp);
    m.def("detect_liquidity_walls", &detect_liquidity_walls_cpp);
    m.def("get_liquidation_heatmap", &get_liquidation_heatmap);
    m.def("find_smart_money_levels", &find_smart_money_levels);
    m.def("analyze_order_book_microstructure", &analyze_order_book_microstructure);
    m.def("analyze_tape_momentum", &analyze_tape_momentum);
}
