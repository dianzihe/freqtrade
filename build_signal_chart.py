# -*- coding: utf-8 -*-
"""生成交互式K线图，展示所有信号触发点。"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "user_data/backtest_results"
L1_CACHE = OUTPUT_DIR / "l1_candles_cache.pkl"
SIGNAL_CSV = OUTPUT_DIR / "信号覆盖率分析数据_v2.csv"


def build_candlestick_chart():
    print("[1] 加载数据...")
    l1 = pd.read_pickle(L1_CACHE)
    sig = pd.read_csv(SIGNAL_CSV, encoding="utf-8-sig")
    sig["signal_time"] = pd.to_datetime(sig["signal_time"], utc=True)

    # 确保 l1 date 是 datetime
    l1["date"] = pd.to_datetime(l1["date"], utc=True)

    print(f"  L1: {len(l1)} 蜡烛, {l1['date'].min()} ~ {l1['date'].max()}")
    print(f"  信号: {len(sig)} 个")

    # ═══════════════════════════════════════════════════════
    # K线数据 (ECharts candlestick 格式: [open, close, low, high])
    # ═══════════════════════════════════════════════════════
    ohlc_data = []
    for _, row in l1.iterrows():
        ts = row["date"].strftime("%Y-%m-%d %H:%M")
        ohlc_data.append([
            float(row["open"]),
            float(row["close"]),
            float(row["low"]),
            float(row["high"]),
        ])

    dates = l1["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    volumes = l1["volume"].fillna(0).tolist()
    spreads = l1["spread_avg"].fillna(0).tolist()

    # ═══════════════════════════════════════════════════════
    # 信号标记数据
    # ═══════════════════════════════════════════════════════
    # 为每个信号找到最近的K线索引
    signal_markers_buy = []  # 买压
    signal_markers_sell = []  # 卖压
    signal_detail = []

    for _, srow in sig.iterrows():
        stime = srow["signal_time"]
        # 找最近的蜡烛
        diff = abs(l1["date"] - stime)
        idx = diff.idxmin()
        candle = l1.iloc[idx]
        cdate = candle["date"].strftime("%Y-%m-%d %H:%M")

        marker = {
            "name": cdate,
            "value": [cdate, float(candle["high"]), float(srow["signal_price"])],
            "symbolSize": 12,
            "itemStyle": {},
        }

        detail = {
            "date": cdate,
            "price": float(srow["signal_price"]),
            "ch1": float(srow["ch1"]),
            "ch2": float(srow["ch2"]),
            "ch3": float(srow["ch3"]),
            "ch4": float(srow["ch4"]),
            "composite": float(srow["composite"]),
            "ch4_dir": str(srow["ch4_dir"]),
            "ret_5m": float(srow["ret_5m"]) * 100,
            "ret_10m": float(srow["ret_10m"]) * 100,
            "max_gain_5m": float(srow["max_gain_5m"]) * 100,
            "max_loss_5m": float(srow["max_loss_5m"]) * 100,
            "max_gain_10m": float(srow["max_gain_10m"]) * 100,
            "max_loss_10m": float(srow["max_loss_10m"]) * 100,
        }

        if str(srow["ch4_dir"]) == "买压":
            marker["itemStyle"]["color"] = "#ef4444"
            signal_markers_buy.append(marker)
        else:
            marker["itemStyle"]["color"] = "#22c55e"
            signal_markers_sell.append(marker)

        signal_detail.append(detail)

    print(f"  买压信号: {len(signal_markers_buy)}, 卖压信号: {len(signal_markers_sell)}")

    # ═══════════════════════════════════════════════════════
    # 价格分段 (按天分片减少ECharts压力)
    # ═══════════════════════════════════════════════════════
    # 将日期和OHLC数据直接注入JSON
    dates_json = json.dumps(dates, ensure_ascii=False)
    ohlc_json = json.dumps(ohlc_data)
    volumes_json = json.dumps(volumes)
    spreads_json = json.dumps(spreads)
    buy_markers_json = json.dumps(signal_markers_buy, ensure_ascii=False)
    sell_markers_json = json.dumps(signal_markers_sell, ensure_ascii=False)
    sig_detail_json = json.dumps(signal_detail, ensure_ascii=False)

    # 分段数据：每6小时一段
    segments = []
    current_seg = {"start": 0, "end": 0, "label": ""}
    seg_size = 6 * 60  # 6 hours of 1-minute candles
    for i, d in enumerate(dates):
        if i == 0:
            current_seg["start"] = 0
            current_seg["label"] = d[:10]  # date only
        elif i % seg_size == 0 or i == len(dates) - 1:
            current_seg["end"] = i
            segments.append(current_seg.copy())
            current_seg = {"start": i, "end": i, "label": d[:10]}
    current_seg["end"] = len(dates) - 1
    segments.append(current_seg)

    segments_json = json.dumps(segments, ensure_ascii=False)

    # ═══════════════════════════════════════════════════════
    # HTML 报告
    # ═══════════════════════════════════════════════════════
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC/USDT K线图 + 微结构信号</title>
<style>
    *{{margin:0;padding:0;box-sizing:border-box;}}
    body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:#0f172a;color:#e2e8f0;}}
    .header{{padding:16px 20px;background:#1e293b;border-bottom:1px solid #334155;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;}}
    .header h1{{font-size:18px;color:#f8fafc;}}
    .header .stats{{font-size:12px;color:#94a3b8;}}
    .header .legend{{display:flex;gap:16px;font-size:12px;}}
    .legend-item{{display:flex;align-items:center;gap:4px;}}
    .dot{{width:10px;height:10px;border-radius:50%;}}
    .main-chart{{width:100%;height:500px;}}
    .sub-charts{{padding:0 20px;}}
    .panel{{background:#1e293b;border-radius:8px;border:1px solid #334155;margin:12px 0;overflow:hidden;}}
    .panel-title{{font-size:14px;font-weight:600;color:#f1f5f9;padding:12px 16px;background:#334155;cursor:pointer;display:flex;align-items:center;gap:8px;}}
    .panel-title .arrow{{transition:transform 0.2s;}}
    .panel-title.collapsed .arrow{{transform:rotate(-90deg);}}
    .signal-gallery{{display:flex;flex-wrap:wrap;gap:8px;padding:12px;max-height:70vh;overflow-y:auto;}}
    .signal-card{{background:#0f172a;border:1px solid #334155;border-radius:6px;padding:8px;cursor:pointer;transition:border-color 0.2s;min-width:180px;flex:1;}}
    .signal-card:hover{{border-color:#60a5fa;}}
    .signal-card .s-time{{font-size:11px;color:#94a3b8;}}
    .signal-card .s-price{{font-size:14px;font-weight:700;}}
    .signal-card .s-ch{{font-size:10px;color:#64748b;margin-top:4px;}}
    .signal-card .s-ret{{font-size:11px;margin-top:2px;}}
    .mini-chart{{width:100%;height:80px;margin-top:4px;}}
    .footer{{padding:16px;color:#64748b;font-size:11px;text-align:center;border-top:1px solid #334155;}}

    /* 滚动条 */
    .signal-gallery::-webkit-scrollbar{{width:6px;}}
    .signal-gallery::-webkit-scrollbar-track{{background:#0f172a;}}
    .signal-gallery::-webkit-scrollbar-thumb{{background:#334155;border-radius:3px;}}
</style>
</head>
<body>
<div class="header">
    <div>
        <h1>BTC/USDT 微结构信号回测</h1>
        <div class="stats">Gate Spot · 1分钟K线 · {len(l1)} 根 · {len(sig)} 个信号</div>
    </div>
    <div class="legend">
        <div class="legend-item"><div class="dot" style="background:#ef4444;"></div> CH4买压 ({len(signal_markers_buy)})</div>
        <div class="legend-item"><div class="dot" style="background:#22c55e;"></div> CH4卖压 ({len(signal_markers_sell)})</div>
    </div>
</div>

<div class="main-chart" id="mainChart"></div>

<div class="sub-charts">
    <div class="panel">
        <div class="panel-title" onclick="togglePanel(this)">
            <span class="arrow">&#9660;</span> 信号缩略图 ({len(sig)} 个)
            <span style="font-weight:400;color:#94a3b8;font-size:12px;margin-left:8px;">点击任意卡片跳转到K线主图对应位置</span>
        </div>
        <div class="signal-gallery" id="signalGallery"></div>
    </div>
</div>

<div class="footer">
    Gate L1 订单簿数据 · 微结构信号模块 (arXiv:2604.20949) · 
    生成: __TIMESTAMP__
</div>

<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<script>
(function() {{
    var dates = {dates_json};
    var ohlc = {ohlc_json};
    var volumes = {volumes_json};
    var spreads = {spreads_json};
    var buyMarkers = {buy_markers_json};
    var sellMarkers = {sell_markers_json};
    var sigDetail = {sig_detail_json};
    var segments = {segments_json};

    // ═════════ 主图 ═════════
    var mainChart = echarts.init(document.getElementById('mainChart'));

    var option = {{
        backgroundColor: '#0f172a',
        animation: false,
        tooltip: {{
            trigger: 'axis',
            axisPointer: {{type:'cross'}},
            backgroundColor:'rgba(30,41,59,0.95)',
            borderColor:'#334155',
            textStyle:{{color:'#e2e8f0',fontSize:12}},
            formatter: function(params) {{
                if (!params || params.length < 2) return '';
                var date = params[0].axisValue;
                // Check if any signal marker is at this date
                var sigInfo = '';
                for (var i = 0; i < params.length; i++) {{
                    if (params[i].seriesName === '买压信号' || params[i].seriesName === '卖压信号') {{
                        var idx = params[i].dataIndex !== undefined ? params[i].dataIndex : -1;
                        if (idx >= 0 && idx < sigDetail.length) {{
                            var s = sigDetail[idx];
                            sigInfo = '<br/><hr style="border-color:#334155"/>' +
                                '<b>' + s.ch4_dir + '</b> 价格: $' + s.price.toFixed(2) + '<br/>' +
                                'CH1: ' + s.ch1.toFixed(3) + ' CH2: ' + s.ch2.toFixed(3) +
                                ' CH3: ' + s.ch3.toFixed(3) + ' CH4: ' + s.ch4.toFixed(3) + '<br/>' +
                                '综合: ' + s.composite.toFixed(3) +
                                ' | 5m收益: ' + s.ret_5m.toFixed(3) + '%' +
                                ' | 10m收益: ' + s.ret_10m.toFixed(3) + '%';
                        }}
                        break;
                    }}
                }}
                var ohlcInfo = '';
                for (var i = 0; i < params.length; i++) {{
                    if (params[i].seriesName === 'K线') {{
                        var d = params[i].data;
                        if (d && d.length >= 4) {{
                            ohlcInfo = '开: ' + d[1].toFixed(2) + ' 收: ' + d[2].toFixed(2) +
                                      ' 低: ' + d[3].toFixed(2) + ' 高: ' + d[4].toFixed(2);
                        }}
                    }}
                }}
                return date + '<br/>' + ohlcInfo + sigInfo;
            }}
        }},
        axisPointer: {{
            link: [{{xAxisIndex: 'all'}}],
            label: {{backgroundColor:'#334155',color:'#e2e8f0'}}
        }},
        grid: [
            {{left:'8%',right:'1%',top:'3%',height:'55%'}},
            {{left:'8%',right:'1%',top:'65%',height:'15%'}},
            {{left:'8%',right:'1%',top:'82%',height:'15%'}},
        ],
        xAxis: [
            {{
                type:'category',data:dates,scale:true,
                boundaryGap:true,axisLine:{{onZero:false,lineStyle:{{color:'#334155'}}}},
                axisLabel:{{color:'#64748b',fontSize:10,formatter:function(v){{return v.substring(5,16);}}}},
                splitLine:{{show:false}},
                min:'dataMin',max:'dataMax',
            }},
            {{
                type:'category',gridIndex:1,data:dates,scale:true,
                boundaryGap:true,axisLine:{{onZero:false,lineStyle:{{color:'#334155'}}}},
                axisLabel:{{show:false}},splitLine:{{show:false}},
                min:'dataMin',max:'dataMax',
            }},
            {{
                type:'category',gridIndex:2,data:dates,scale:true,
                boundaryGap:true,axisLine:{{onZero:false,lineStyle:{{color:'#334155'}}}},
                axisLabel:{{color:'#64748b',fontSize:10,formatter:function(v){{return v.substring(5,16);}}}},
                splitLine:{{show:false}},
                min:'dataMin',max:'dataMax',
            }},
        ],
        yAxis: [
            {{
                scale:true,splitArea:{{show:true,areaStyle:{{color:['rgba(30,41,59,0.4)','rgba(15,23,42,0.4)']}}}},
                axisLine:{{lineStyle:{{color:'#334155'}}}},
                axisLabel:{{color:'#94a3b8',fontSize:10,formatter:'${{value}}'}},
                splitLine:{{lineStyle:{{color:'#1e293b'}}}}
            }},
            {{
                scale:true,gridIndex:1,splitNumber:3,
                axisLine:{{lineStyle:{{color:'#334155'}}}},
                axisLabel:{{color:'#64748b',fontSize:9}},
                splitLine:{{lineStyle:{{color:'#1e293b'}}}}
            }},
            {{
                scale:true,gridIndex:2,splitNumber:3,
                axisLine:{{lineStyle:{{color:'#334155'}}}},
                axisLabel:{{color:'#64748b',fontSize:9}},
                splitLine:{{lineStyle:{{color:'#1e293b'}}}}
            }},
        ],
        dataZoom: [
            {{
                type:'inside',xAxisIndex:[0,1,2],start:0,end:100,
                zoomOnMouseWheel:true,moveOnMouseMove:true,
            }},
            {{
                type:'slider',xAxisIndex:[0,1,2],start:0,end:100,
                height:20,bottom:0,
                borderColor:'#334155',backgroundColor:'#1e293b',
                fillerColor:'rgba(96,165,250,0.2)',
                handleStyle:{{color:'#60a5fa'}},
                textStyle:{{color:'#94a3b8'}},
            }},
        ],
        series: [
            {{
                name:'K线',type:'candlestick',data:ohlc,
                xAxisIndex:0,yAxisIndex:0,
                itemStyle:{{
                    color:'#ef4444',color0:'#22c55e',
                    borderColor:'#ef4444',borderColor0:'#22c55e',
                }},
                markPoint: {{
                    symbol:'pin',symbolSize:40,
                    data: buyMarkers.concat(sellMarkers),
                }},
            }},
            {{
                name:'买压信号',type:'scatter',data:buyMarkers,
                xAxisIndex:0,yAxisIndex:0,
                symbol:'triangle',symbolSize:14,symbolRotate:180,
                itemStyle:{{color:'#ef4444',borderColor:'#fff',borderWidth:1}},
                emphasis:{{scale:2}},
            }},
            {{
                name:'卖压信号',type:'scatter',data:sellMarkers,
                xAxisIndex:0,yAxisIndex:0,
                symbol:'triangle',symbolSize:14,
                itemStyle:{{color:'#22c55e',borderColor:'#fff',borderWidth:1}},
                emphasis:{{scale:2}},
            }},
            {{
                name:'成交量',type:'bar',xAxisIndex:1,yAxisIndex:1,
                data:volumes.map(function(v,i){{
                    var close = ohlc[i] ? ohlc[i][1] : 0;
                    var open = ohlc[i] ? ohlc[i][0] : 0;
                    return {{value:v,itemStyle:{{color:close>=open?'#ef4444':'#22c55e'}}}};
                }}),
            }},
            {{
                name:'价差',type:'line',xAxisIndex:2,yAxisIndex:2,
                data:spreads,
                lineStyle:{{color:'#f59e0b',width:1}},
                areaStyle:{{color:'rgba(245,158,11,0.1)'}},
                symbol:'none',
            }},
        ],
    }};
    mainChart.setOption(option);

    // ═════════ 信号缩略图 ═════════
    var gallery = document.getElementById('signalGallery');

    // 为每个信号生成小图
    var sigCount = sigDetail.length;
    for (var i = 0; i < sigCount; i++) {{
        var s = sigDetail[i];

        // 生成小K线 (前后30分钟 + 信号)
        var miniData = [];
        var dateIdx = dates.indexOf(s.date);
        if (dateIdx >= 0) {{
            var start = Math.max(0, dateIdx - 20);
            var end = Math.min(dates.length, dateIdx + 35);
            for (var j = start; j < end; j++) {{
                if (j < ohlc.length) {{
                    miniData.push(ohlc[j]);
                }}
            }}
        }}

        var miniJson = JSON.stringify(miniData);
        var containerId = 'mini_' + i;

        var card = document.createElement('div');
        card.className = 'signal-card';
        card.onclick = function(dateStr) {{
            return function() {{
                // 跳转到主图对应位置
                var idx = dates.indexOf(dateStr);
                if (idx >= 0) {{
                    var total = dates.length;
                    var pct = Math.max(0, (idx - 30) / total * 100);
                    mainChart.dispatchAction({{
                        type:'dataZoom',
                        startValue:dates[Math.max(0,idx-30)],
                        endValue:dates[Math.min(total-1,idx+60)],
                    }});
                }}
            }};
        }}(s.date);

        var ch4Color = s.ch4_dir === '买压' ? '#ef4444' : '#22c55e';
        var ret5Color = s.ret_5m >= 0 ? '#ef4444' : '#22c55e';
        var ret10Color = s.ret_10m >= 0 ? '#ef4444' : '#22c55e';

        card.innerHTML =
            '<div class="s-time">' + s.date.substring(5) + '</div>' +
            '<div class="s-price" style="color:' + ch4Color + '">$' + s.price.toFixed(0) +
                ' <span style="font-size:10px;color:#94a3b8;">' + s.ch4_dir + '</span></div>' +
            '<div class="s-ch">CH1:' + s.ch1.toFixed(2) + ' CH2:' + s.ch2.toFixed(2) +
                ' CH3:' + s.ch3.toFixed(2) + ' CH4:' + s.ch4.toFixed(2) + '</div>' +
            '<div class="s-ret">' +
                '<span style="color:' + ret5Color + ';">5m: ' + (s.ret_5m >= 0 ? '+' : '') + s.ret_5m.toFixed(3) + '%</span> ' +
                '<span style="color:' + ret10Color + ';">10m: ' + (s.ret_10m >= 0 ? '+' : '') + s.ret_10m.toFixed(3) + '%</span>' +
            '</div>' +
            '<div class="mini-chart" id="' + containerId + '"></div>';

        gallery.appendChild(card);

        // 延迟渲染小图
        setTimeout(function(cid, data, sigPrice) {{
            var el = document.getElementById(cid);
            if (!el) return;

            var mini = echarts.init(el);
            var minP = Infinity, maxP = -Infinity;
            for (var k = 0; k < data.length; k++) {{
                if (data[k][3] < minP) minP = data[k][3];
                if (data[k][4] > maxP) maxP = data[k][4];
            }}
            if (sigPrice < minP) minP = sigPrice;
            if (sigPrice > maxP) maxP = sigPrice;
            var pr = maxP - minP || 1;

            var sigLineData = data.map(function(d){{return sigPrice;}});

            mini.setOption({{
                backgroundColor:'transparent',
                grid:{{left:2,right:2,top:2,bottom:2}},
                xAxis:{{show:false,data:data.map(function(_,i){{return i;}})}},
                yAxis:{{show:false,min:minP-pr*0.05,max:maxP+pr*0.05}},
                series: [
                    {{
                        type:'candlestick',data:data,
                        itemStyle:{{
                            color:'#ef4444',color0:'#22c55e',
                            borderColor:'#ef4444',borderColor0:'#22c55e',
                            borderWidth:0.5,
                        }},
                        barWidth:'60%',
                    }},
                    {{
                        type:'line',data:sigLineData,
                        lineStyle:{{color:ch4Color,width:1,type:'dashed'}},
                        symbol:'none',silent:true,
                    }}
                ]
            }});
        }}, 100 + i * 10, containerId, miniData, s.price);
    }}

    // 响应式
    window.addEventListener('resize', function(){{mainChart.resize();}});
}})();

function togglePanel(titleEl) {{
    titleEl.classList.toggle('collapsed');
    var gallery = titleEl.nextElementSibling;
    if (gallery.style.display === 'none') {{
        gallery.style.display = 'flex';
    }} else {{
        gallery.style.display = 'none';
    }}
}}
</script>
</body>
</html>"""

    # Replace timestamp
    html = html.replace(
        "__TIMESTAMP__",
        pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
    )

    output_path = OUTPUT_DIR / "K线信号展示.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"\n[2] 报告已保存: {output_path}")
    print(f"  文件大小: {output_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build_candlestick_chart()
