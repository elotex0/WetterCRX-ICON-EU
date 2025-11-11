import sys
import cfgrib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os
from adjustText import adjust_text
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from scipy.interpolate import RegularGridInterpolator
from zoneinfo import ZoneInfo
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "output"
output_dir = sys.argv[2]      # z.B. "output/maps"
var_type = sys.argv[3]        # 't2m', 'ww', 'tp', 'tp_acc', 'cape_ml', 'dbz_cmax'
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Geo-Daten
# ------------------------------
cities = pd.DataFrame({
    'name': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Dresden', 'Stuttgart', 'Düsseldorf',
             'Nürnberg', 'Erfurt', 'Leipzig', 'Bremen', 'Saarbrücken', 'Hannover'],
    'lat': [52.52, 53.55, 48.14, 50.94, 50.11, 51.05, 48.78, 51.23,
            49.45, 50.98, 51.34, 53.08, 49.24, 52.37],
    'lon': [13.40, 9.99, 11.57, 6.96, 8.68, 13.73, 9.18, 6.78,
            11.08, 11.03, 12.37, 8.80, 6.99, 9.73]
})

eu_cities = pd.DataFrame({
    'name': [
        'Berlin', 'Oslo', 'Warschau',
        'Lissabon', 'Madrid', 'Rom',
        'Ankara', 'Helsinki', 'Reykjavik',
        'London', 'Paris'
    ],
    'lat': [
        52.52, 59.91, 52.23,
        38.72, 40.42, 41.90,
        39.93, 60.17, 64.13,
        51.51, 48.85
    ],
    'lon': [
        13.40, 10.75, 21.01,
        -9.14, -3.70, 12.48,
        32.86, 24.94, -21.82,
        -0.13, 2.35
    ]
})

ignore_codes = {4}

# ------------------------------
# WW-Farben
# ------------------------------
ww_colors_base = {
    0: "#FFFFFF", 1: "#D3D3D3", 2: "#A9A9A9", 3: "#696969",
    45: "#FFFF00", 48: "#FFD700",
    56: "#FFA500", 57: "#C06A00",
    51: "#00FF00", 53: "#00C300", 55: "#009700",
    61: "#00FF00", 63: "#00C300", 65: "#009700",
    80: "#00FF00", 81: "#00C300", 82: "#009700",
    66: "#FF6347", 67: "#8B0000",
    71: "#ADD8E6", 73: "#6495ED", 75: "#00008B",
    85: "#ADD8E6", 86: "#6495ED",
    77: "#ADD8E6",
    95: "#FF77FF", 96: "#C71585", 99: "#C71585"
}
ww_categories = {
    "Bewölkung": [0, 1 , 2, 3],
    "Nebel": [45],
    "Schneeregen": [56, 57],
    "Regen": [61, 63, 65],
    "gefr. Regen": [66, 67],
    "Schnee": [71, 73, 75],
    "Gewitter": [95,96],
}

# ------------------------------
# Temperatur-Farben
# ------------------------------
t2m_bounds = list(range(-36, 50, 2))
t2m_colors = LinearSegmentedColormap.from_list(
    "t2m_smoooth",
    [
    "#F675F4", "#F428E9", "#B117B5", "#950CA2", "#640180",
    "#3E007F", "#00337E", "#005295", "#1292FF", "#49ACFF",
    "#8FCDFF", "#B4DBFF", "#B9ECDD", "#88D4AD", "#07A125",
    "#3FC107", "#9DE004", "#E7F700", "#F3CD0A", "#EE5505",
    "#C81904", "#AF0E14", "#620001", "#C87879", "#FACACA",
    "#E1E1E1", "#6D6D6D"
    ],
N=len(t2m_bounds)
)
t2m_norm = BoundaryNorm(t2m_bounds, ncolors=len(t2m_bounds))

# ------------------------------
# Niederschlags-Farben 1h (tp)
# ------------------------------

# ------------------------------
# Aufsummierter Niederschlag (tp_acc)
# ------------------------------
tp_acc_bounds = [0.1, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                 125, 150, 175, 200, 250, 300, 400, 500]
tp_acc_colors = ListedColormap([
    "#B4D7FF","#75BAFF","#349AFF","#0582FF","#0069D2",
    "#003680","#148F1B","#1ACF06","#64ED07","#FFF32B",
    "#E9DC01","#F06000","#FF7F26","#FFA66A","#F94E78",
    "#F71E53","#BE0000","#880000","#64007F","#C201FC",
    "#DD66FE","#EBA6FF","#F9E7FF","#D4D4D4","#969696"
])
tp_acc_norm = mcolors.BoundaryNorm(tp_acc_bounds, tp_acc_colors.N)

# ------------------------------
# Windböen-Farben
# ------------------------------
wind_bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300]
wind_colors = ListedColormap([
    "#68AD05", "#8DC00B", "#B1D415", "#D5E81C", "#FBFC22",
    "#FAD024", "#F9A427", "#FC7929", "#FB4D2B", "#EA2B57",
    "#FB22A5", "#FC22CE", "#FC22F5", "#FC62F8", "#FD80F8",
    "#FFBFFC", "#FEDFFE", "#FEFFFF", "#E1E0FF", "#C3C3FF",
    "#A5A5FF", "#A5A5FF", "#6868FE"
])
wind_norm = mcolors.BoundaryNorm(wind_bounds, wind_colors.N)

#-------------------------------
# Schneehöhen-Farben
#------------------------------
snow_bounds = [0, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 400]  # in cm
snow_colors = ListedColormap([
        "#F8F8F8", "#DCDBFA", "#AAA9C8", "#75BAFF", "#349AFF", "#0682FF",
        "#0069D2", "#004F9C", "#01327F", "#4B007F", "#64007F", "#9101BB",
        "#C300FC", "#D235FF", "#EBA6FF", "#F4CEFF", "#FAB2CA", "#FF9798",
        "#FE6E6E", "#DF093F", "#BE0000", "#A40000", "#880000"
    ])
snow_norm = mcolors.BoundaryNorm(snow_bounds, snow_colors.N)

# ------------------------------
# Schneefallgrenze (SNOWLMT)
# ------------------------------

snowfall_bounds = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
snowfall_colors = ListedColormap([
    "#FF00A6", "#D900FF", "#8C00FF", "#0008FF", "#0099FF",
    "#00F2FF", "#1AFF00", "#FFFB00", "#FFBF00", "#FFA600",
    "#FF6F00", "#930000", 
])

snowfall_norm = mcolors.BoundaryNorm(snowfall_bounds, snowfall_colors.N)

# ------------------------------
# Luftdruck
# ------------------------------

# Luftdruck-Farben (kontinuierlicher Farbverlauf für 45 Bins)
pmsl_bounds_colors = list(range(912, 1070, 4))  # Alle 4 hPa (45 Bins)
pmsl_colors = LinearSegmentedColormap.from_list(
    "pmsl_smooth",
    [
       "#FF6DFF", "#C418C4", "#950CA2", "#5A007D", "#3D007F",
       "#00337E", "#0472CB", "#4FABF8", "#A3D4FF", "#79DAAD",
       "#07A220", "#3EC008", "#9EE002", "#F3FC01", "#F19806",
       "#F74F11", "#B81212", "#8C3234", "#C87879", "#F9CBCD",
       "#E2E2E2"

    ],
    N=len(pmsl_bounds_colors)  # Genau 45 Farben für 45 Bins
)
pmsl_norm = BoundaryNorm(pmsl_bounds_colors, ncolors=len(pmsl_bounds_colors))

# ------------------------------
# Geopotenzial
# ------------------------------

geo_bounds = list(range(4800, 6000, 40))
geo_colors = LinearSegmentedColormap.from_list(
    "geo_smooth",
    [
        "#530155", "#6F1171", "#871D89", "#9E2C9E", "#B73AB2", "#CB49CD", "#9D3AD2",
        "#6C2ECF", "#3B20C5", "#0B12B8", "#0D2FC4", "#124FC4", "#136AB7", "#1889C1",
        "#149A99", "#06B16F", "#10BA4D", "#09CC28", "#FECC0B", "#FEB906", "#F5A40A",
        "#F09006", "#E38500", "#EB6C01", "#E45C04", "#DC4A01", "#DB3600", "#D42601",
        "#C31700", "#CB0003", "#4E0703"
    ],
    N=len(geo_bounds)
)
geo_norm = BoundaryNorm(geo_bounds, ncolors=len(geo_bounds))


# ------------------------------
# Kartenparameter
# ------------------------------
FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT = FIG_W_PX / TOP_AREA_PX

# Bounding Box Deutschland (fix, keine GeoJSON nötig)
extent = [5, 16, 47, 56]

extent_eu = [-23.5, 45.0, 29.5, 68.4]

# ------------------------------
# WW-Legende Funktion
# ------------------------------
def add_ww_legend_bottom(fig, ww_categories, ww_colors_base):
    legend_height = 0.12
    legend_ax = fig.add_axes([0.05, 0.01, 0.9, legend_height])
    legend_ax.axis("off")
    for i, (label, codes) in enumerate(ww_categories.items()):
        n_colors = len(codes)
        block_width = 1.0 / len(ww_categories)
        gap = 0.05 * block_width
        x0 = i * block_width
        x1 = (i + 1) * block_width
        inner_width = x1 - x0 - gap
        color_width = inner_width / n_colors
        for j, c in enumerate(codes):
            color = ww_colors_base.get(c, "#FFFFFF")
            legend_ax.add_patch(mpatches.Rectangle((x0 + j * color_width, 0.3),
                                                  color_width, 0.6,
                                                  facecolor=color, edgecolor='black'))
        legend_ax.text((x0 + x1)/2, 0.05, label, ha='center', va='bottom', fontsize=10)

# ------------------------------
# Dateien durchgehen
# ------------------------------
for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".grib2"):
        continue
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    # Daten je Typ
    if var_type == "t2m":
        if "t2m" not in ds:
            print(f"Keine t2m in {filename}")
            continue
        data = ds["t2m"].values - 273.15
    elif var_type == "t850":
        if "t" not in ds:
            print(f"Keine t in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["t"].values - 273.15
    elif var_type == "t850_eu":
        if "t" not in ds:
            print(f"Keine t in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["t"].values - 273.15
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn.lower() in ["ww","weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            continue
        data = ds[varname].values
    elif var_type == "tp_acc":
        tp_var = next((vn for vn in ["tp","tot_prec"] if vn in ds), None)
        if tp_var is None:
            print(f"Keine Niederschlagsvariable in {filename}")
            continue
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        tp_all = ds[tp_var].values
        if tp_all.ndim == 1:
            ny, nx = len(lat), len(lon)
            tp_all = tp_all.reshape(ny, nx)
        elif tp_all.ndim == 3:
            data = tp_all[3]-tp_all[0] if tp_all.shape[0]>1 else tp_all[0]
        else:
            data = tp_all
        lon2d, lat2d = np.meshgrid(lon, lat)
        data[data < 0.1] = np.nan
    elif var_type == "cape_ml":
        if "CAPE_ML" not in ds:
            print(f"Keine CAPE_ML-Variable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["CAPE_ML"].values
        data[data<0]=np.nan
    elif var_type == "dbz_cmax":
        if "DBZ_CMAX" not in ds:
            print(f"Keine DBZ_CMAX in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["DBZ_CMAX"].values[0,:,:]
    elif var_type == "wind":
        if "fg10" not in ds:
            print(f"Keine passende Windvariable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["fg10"].values
        data[data < 0] = np.nan
        data = data * 3.6  # m/s → km/h
    elif var_type == "wind_eu":
        if "fg10" not in ds:
            print(f"Keine passende Windvariable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["fg10"].values
        data[data < 0] = np.nan
        data = data * 3.6  # m/s → km/h
    elif var_type == "snow":
        if "sde" not in ds:
            print(f"Keine sde-Variable in {filename}")
            continue
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        lon2d, lat2d = np.meshgrid(lon, lat)
        data = ds["sde"].values
        data[data < 0] = np.nan
        data = data * 100  # in cm umrechnen
    elif var_type == "cloud":
        if "CLCT" not in ds:
            print(f"Keine CLCT-Variable in {filename}")
            continue
        data = ds["CLCT"].values
        data[data < 0] = np.nan
    elif var_type == "twater":
        if "TWATER" not in ds:
            print(f"Keine TWATER-Variable in {filename}")
            continue
        data = ds["TWATER"].values
        data[data < 0] = np.nan
    elif var_type == "snowfall":
        if "SNOWLMT" not in ds:
            print(f"Keine SNOWLMT-Variable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["SNOWLMT"].values
        data[data < 0] = np.nan
    elif var_type == "pmsl":
        if "prmsl" not in ds:
            print(f"Keine prmsl-Variable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["prmsl"].values / 100
        data[data < 0] = np.nan
    elif var_type == "pmsl_eu":
        if "prmsl" not in ds:
            print(f"Keine prmsl-Variable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["prmsl"].values / 100
        data[data < 0] = np.nan
    elif var_type == "geo_eu":
        if "z" not in ds:
            print(f"Keine z-Variable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["z"].values / 9.80665
        data[data < 0] = np.nan
    else:
        print(f"Unbekannter var_type {var_type}")
        continue

    if data.ndim==3:
        data=data[0]

    lon = ds["longitude"].values
    lat = ds["latitude"].values
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None

    if "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0])
        valid_time_utc = run_time_utc + step
    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))

    # --------------------------
    # Figure (Deutschland oder Europa)
    # --------------------------
    if var_type in ["pmsl_eu", "wind_eu", "t850_eu", "geo_eu"]:
        scale = 0.9
        fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
        shift_up = 0.02
        ax = fig.add_axes([0.0, BOTTOM_AREA_PX / FIG_H_PX + shift_up, 1.0, TOP_AREA_PX / FIG_H_PX],
                        projection=ccrs.PlateCarree())
        ax.set_extent(extent_eu)
        ax.set_axis_off()
        ax.set_aspect('auto')
    else:
        scale = 0.9
        fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
        shift_up = 0.02
        ax = fig.add_axes([0.0, BOTTOM_AREA_PX / FIG_H_PX + shift_up, 1.0, TOP_AREA_PX / FIG_H_PX],
                        projection=ccrs.PlateCarree())
        ax.set_extent(extent)
        ax.set_axis_off()
        ax.set_aspect('auto')


    if var_type in ["pmsl_eu", "wind_eu", "t850_eu", "geo_eu"]:
        target_res = 0.1   # gröber für Europa (~11 km)
        lon_min, lon_max, lat_min, lat_max = extent_eu
        buffer = target_res * 20
        nx = int(round(lon_max - lon_min) / target_res) + 1
        ny = int(round(lat_max - lat_min) / target_res) + 1
        lon_new = np.linspace(lon_min - buffer, lon_max + buffer, nx + 15)
        lat_new = np.linspace(lat_min - buffer, lat_max + buffer, ny + 15)
        lon2d_new, lat2d_new = np.meshgrid(lon_new, lat_new)

    else:
        target_res = 0.025  # feiner für Deutschland (~2.8 km)
        lon_min, lon_max, lat_min, lat_max = extent
        lon_new = np.arange(lon_min, lon_max + target_res, target_res)
        lat_new = np.arange(lat_min, lat_max + target_res, target_res)
        lon2d_new, lat2d_new = np.meshgrid(lon_new, lat_new)


    # Nur interpolieren, wenn Daten reguläres 2D-Gitter haben
    if lon.ndim == 1 and lat.ndim == 1 and data.ndim == 2:
        try:
            if var_type == "ww":
                # 🧱 Kategorische Interpolation: nearest-neighbor
                interp_func = RegularGridInterpolator(
                    (lat[::-1], lon),
                    data[::-1, :],
                    method="nearest",          # <--- WICHTIG
                    bounds_error=False,
                    fill_value=np.nan
                )
            else:
                # 🌈 Kontinuierliche Interpolation: linear
                interp_func = RegularGridInterpolator(
                    (lat[::-1], lon),
                    data[::-1, :],
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan
                )

            pts = np.array([lat2d_new.ravel(), lon2d_new.ravel()]).T
            data = interp_func(pts).reshape(lat2d_new.shape)
            lon, lat = lon_new, lat_new
            lon2d, lat2d = lon2d_new, lat2d_new
        except Exception as e:
            print(f"Interpolation übersprungen ({e})")

    # Plot
    if var_type == "t2m":
        im = ax.pcolormesh(lon, lat, data, cmap=t2m_colors, norm=t2m_norm, shading="auto")
    elif var_type == "t850":
        im = ax.pcolormesh(lon, lat, data, cmap=t2m_colors, norm=t2m_norm, shading="auto")
    elif var_type == "t850_eu":
        im = ax.pcolormesh(lon, lat, data, cmap=t2m_colors, norm=t2m_norm, shading="auto")
    elif var_type == "ww":
        valid_mask = np.isfinite(data)
        codes = np.unique(data[valid_mask]).astype(int)
        codes = [c for c in codes if c in ww_colors_base and c not in ignore_codes]
        codes.sort()
        cmap = ListedColormap([ww_colors_base[c] for c in codes])
        code2idx = {c: i for i, c in enumerate(codes)}
        idx_data = np.full_like(data, fill_value=np.nan, dtype=float)
        for c,i in code2idx.items():
            idx_data[data==c]=i
        im = ax.pcolormesh(lon, lat, idx_data, cmap=cmap, vmin=-0.5, vmax=len(codes)-0.5, shading="auto")
    elif var_type == "tp_acc":
        im = ax.pcolormesh(lon2d, lat2d, data, cmap=tp_acc_colors, norm=tp_acc_norm, shading="auto")
    elif var_type == "wind":
        im = ax.pcolormesh(lon, lat, data, cmap=wind_colors, norm=wind_norm, shading="auto")
    elif var_type == "wind_eu":
        im = ax.pcolormesh(lon, lat, data, cmap=wind_colors, norm=wind_norm, shading="auto")
    elif var_type == "snow":
        im = ax.pcolormesh(lon2d, lat2d, data, cmap=snow_colors, norm=snow_norm, shading="auto")
    elif var_type == "snowfall":
        im = ax.pcolormesh(lon, lat, data, cmap=snowfall_colors, norm=BoundaryNorm(snowfall_bounds, snowfall_colors.N), shading="auto")
    elif var_type == "pmsl":
        # --- Luftdruck auf Meereshöhe (Deutschland) ---
        im = ax.pcolormesh(lon, lat, data, cmap=pmsl_colors, norm=pmsl_norm, shading="auto")
        data_hpa = data  # Daten liegen bereits in hPa vor

        # Haupt-Isobaren (alle 4 hPa)
        main_levels = list(range(912, 1070, 4))
        # Feine Isobaren (alle 1 hPa)
        fine_levels = list(range(912, 1070, 1))

        # Nur Levels zeichnen, die im Datenbereich liegen
        main_levels = [lev for lev in main_levels if data_hpa.min() <= lev <= data_hpa.max()]
        fine_levels = [lev for lev in fine_levels if data_hpa.min() <= lev <= data_hpa.max()]

        # Feine Isobaren (weiß, dünn, leicht transparent)
        ax.contour(
            lon, lat, data_hpa,
            levels=fine_levels,
            colors='gray', linewidths=0.5, alpha=0.4
        )

        # Haupt-Isobaren (weiß, etwas dicker)
        cs_main = ax.contour(
            lon, lat, data_hpa,
            levels=main_levels,
            colors='white', linewidths=0.8, alpha=0.9
        )

        # Isobaren-Beschriftung (Zahlen direkt auf Linien)
        ax.clabel(cs_main, inline=True, fmt='%d', fontsize=9, colors='black')

        # --- Extremwerte (Tief & Hoch) markieren, aber nur wenn im Extent ---
        min_idx = np.unravel_index(np.nanargmin(data_hpa), data_hpa.shape)
        max_idx = np.unravel_index(np.nanargmax(data_hpa), data_hpa.shape)
        min_val = data_hpa[min_idx]
        max_val = data_hpa[max_idx]

        lon_min, lon_max, lat_min, lat_max = extent

        # Tiefdruckzentrum (blauer Wert)
        lon_minpt, lat_minpt = lon[min_idx[1]], lat[min_idx[0]]
        if lon_min <= lon_minpt <= lon_max and lat_min <= lat_minpt <= lat_max:
            ax.text(
                lon_minpt, lat_minpt,
                f"{min_val:.0f}",
                color='blue', fontsize=12, fontweight='bold',
                ha='center', va='center',
                transform=ccrs.PlateCarree(),
                clip_on=True,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')]
            )

        # Hochdruckzentrum (roter Wert)
        lon_maxpt, lat_maxpt = lon[max_idx[1]], lat[max_idx[0]]
        if lon_min <= lon_maxpt <= lon_max and lat_min <= lat_maxpt <= lat_max:
            ax.text(
                lon_maxpt, lat_maxpt,
                f"{max_val:.0f}",
                color='red', fontsize=12, fontweight='bold',
                ha='center', va='center',
                transform=ccrs.PlateCarree(),
                clip_on=True,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')]
            )



    elif var_type == "pmsl_eu":
            # Schnellere Variante ohne adjust_text und smoothing
            im = ax.pcolormesh(lon, lat, data, cmap=pmsl_colors, norm=pmsl_norm, shading="auto")
            data_hpa = data  # data schon in hPa
            main_levels = list(range(912, 1070, 4))
            cs = ax.contour(lon, lat, data_hpa, levels=main_levels,
                            colors='white', linewidths=0.8, alpha=0.9)
            ax.clabel(cs, inline=True, fmt='%d', fontsize=9, colors='black')

            low_levels = list(range(912, 1070, 1))
            ax.contour(lon, lat, data_hpa, levels=low_levels,
                             colors='gray', linewidths=0.5, alpha=0.4)

            # Min/Max-Druck markieren (optional)
            min_idx = np.unravel_index(np.nanargmin(data_hpa), data_hpa.shape)
            max_idx = np.unravel_index(np.nanargmax(data_hpa), data_hpa.shape)

            ax.text(
                lon[min_idx[1]], lat[min_idx[0]],
                f"T\n{data_hpa[min_idx]:.0f}",
                color='blue', fontsize=11, fontweight='bold',
                ha='center', va='center',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')]
            )

            ax.text(
                lon[max_idx[1]], lat[max_idx[0]],
                f"H\n{data_hpa[max_idx]:.0f}",
                color='red', fontsize=11, fontweight='bold',
                ha='center', va='center',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')]
            )

    elif var_type == "geo_eu":
            im = ax.pcolormesh(lon, lat, data, cmap=geo_colors, norm=geo_norm, shading="auto")
            data_geo = data  # in m # data schon in hPa
            main_levels = list(range(4800, 6000, 40))
            cs = ax.contour(lon, lat, data_geo, levels=main_levels,
                            colors='white', linewidths=0.8, alpha=0.9)
            ax.clabel(cs, inline=True, fmt='%d', fontsize=9, colors='black')

            low_levels = list(range(4800, 6000, 10))
            ax.contour(lon, lat, data_geo, levels=low_levels,
                            colors='gray', linewidths=0.5, alpha=0.4)

            # Min/Max-Druck markieren (optional)
            min_idx = np.unravel_index(np.nanargmin(data_geo), data_geo.shape)
            max_idx = np.unravel_index(np.nanargmax(data_geo), data_geo.shape)

            ax.text(
                lon[min_idx[1]], lat[min_idx[0]],
                f"{data_geo[min_idx]:.0f}",
                color='white', fontsize=11, fontweight='bold',
                ha='center', va='center',
                transform=ccrs.PlateCarree(),
                clip_on=True,
                path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')]
            )

            ax.text(
                lon[max_idx[1]], lat[max_idx[0]],
                f"{data_geo[max_idx]:.0f}",
                color='white', fontsize=11, fontweight='bold',
                ha='center', va='center',
                transform=ccrs.PlateCarree(),
                clip_on=True,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')]
            )


    # ------------------------------
    # Grenzen & Städte
    # ------------------------------

    if var_type in ["pmsl_eu", "wind_eu", "t850_eu", "geo_eu"]:
        # 🌍 Europa: nur Ländergrenzen + europäische Städte
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="black", linewidth=0.7)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black", linewidth=0.7)

        for _, city in eu_cities.iterrows():
            ax.plot(city["lon"], city["lat"], "o", markersize=6,
                    markerfacecolor="black", markeredgecolor="white",
                    markeredgewidth=1.5, zorder=5)
            txt = ax.text(city["lon"] + 0.3, city["lat"] + 0.3, city["name"],
                          fontsize=9, color="black", weight="bold", zorder=6)
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])

    else:
        # 🇩🇪 Deutschland: Bundesländer, Grenzen und Städte
        ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="#2C2C2C", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black")

        for _, city in cities.iterrows():
            ax.plot(city["lon"], city["lat"], "o", markersize=6,
                    markerfacecolor="black", markeredgecolor="white",
                    markeredgewidth=1.5, zorder=5)
            txt = ax.text(city["lon"] + 0.1, city["lat"] + 0.1, city["name"],
                          fontsize=9, color="black", weight="bold", zorder=6)
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])

    # Rahmen um Karte
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    fill=False, color="black", linewidth=2))


    # Legende
    legend_h_px = 50
    legend_bottom_px = 45
    if var_type in ["t2m","tp_acc","cape_ml","dbz_cmax","wind","snow", "cloud", "twater", "snowfall", "pmsl", "pmsl_eu", "wind_eu", "t850_eu", "t850", "geo_eu"]:
        bounds = t2m_bounds if var_type=="t2m" else tp_acc_bounds if var_type=="tp_acc" else wind_bounds if var_type=="wind" else snow_bounds if var_type=="snow" else snowfall_bounds if var_type=="snowfall" else pmsl_bounds_colors if var_type=="pmsl" else pmsl_bounds_colors if var_type=="pmsl_eu" else wind_bounds if var_type=="wind_eu" else t2m_bounds if var_type=="t850_eu" else t2m_bounds if var_type=="t850" else geo_bounds
        cbar_ax = fig.add_axes([0.03, legend_bottom_px / FIG_H_PX, 0.94, legend_h_px / FIG_H_PX])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=bounds)
        cbar.ax.tick_params(colors="black", labelsize=7)
        cbar.outline.set_edgecolor("black")
        cbar.ax.set_facecolor("white")

        # Für pmsl nur jeden 10. hPa Tick beschriften
        if var_type=="pmsl":
            tick_labels = [str(tick) if tick % 8 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type=="pmsl_eu":
            tick_labels = [str(tick) if tick % 8 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type == "t2m":
            tick_labels = [str(tick) if tick % 4 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type == "t850":
            tick_labels = [str(tick) if tick % 4 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type == "t850_eu":
            tick_labels = [str(tick) if tick % 4 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type == "geo_eu":
            tick_labels = [str(tick) if tick % 80 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)

        if var_type=="tp_acc":
            cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in tp_acc_bounds])
        if var_type=="snow":
            cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in snow_bounds])
    else:
        add_ww_legend_bottom(fig, ww_categories, ww_colors_base)

    # Footer
    footer_ax = fig.add_axes([0.0, (legend_bottom_px + legend_h_px)/FIG_H_PX, 1.0,
                              (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px)/FIG_H_PX])
    footer_ax.axis("off")
    footer_texts = {
        "ww": "Signifikantes Wetter",
        "t2m": "Temperatur 2m (°C)",
        "t850": "Temperatur 850hPa (°C)",
        "tp_acc": "Akkumulierter Niederschlag (mm)",
        "cape_ml": "CAPE-Index (J/kg)",
        "cloud": "Gesamtbewölkung (%)",
        "wind": "Windböen (km/h)",
        "snow": "Schneehöhe (cm)",
        "twater": "Gesamtwassergehalt (mm)",
        "snowfall": "Schneefallgrenze (m)",
        "pmsl": "Luftdruck auf Meereshöhe (hPa)",
        "pmsl_eu": "Luftdruck auf Meereshöhe (hPa), Europa",
        "wind_eu": "Windböen (km/h), Europa",
        "t850_eu": "Temperatur 850hPa (°C), Europa",
        "geo_eu": "Geopotentielle Höhe 500hPa (m), Europa",
    }

    left_text = footer_texts.get(var_type, var_type) + \
                f"\nICON-EU ({pd.to_datetime(run_time_utc).hour:02d}z), Deutscher Wetterdienst" \
                if run_time_utc is not None else \
                footer_texts.get(var_type, var_type) + "\nICON-EU (??z), Deutscher Wetterdienst"

    footer_ax.text(0.01, 0.85, left_text, fontsize=12, fontweight="bold", va="top", ha="left")
    footer_ax.text(0.734, 0.92, "Prognose für:", fontsize=12, va="top", ha="left", fontweight="bold")
    footer_ax.text(0.99, 0.68, f"{valid_time_local:%d.%m.%Y %H:%M} Uhr",
                   fontsize=12, va="top", ha="right", fontweight="bold")

    # Speichern
    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}.png"
    plt.savefig(os.path.join(output_dir, outname), dpi=100, bbox_inches=None, pad_inches=0)
    plt.close()
