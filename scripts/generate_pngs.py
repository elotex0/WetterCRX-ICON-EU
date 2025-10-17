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
# CAPE-Farben
# ------------------------------
cape_bounds = [0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
cape_colors = ListedColormap([
    "#676767", "#006400", "#008000", "#00CC00", "#66FF00", "#FFFF00", 
    "#FFCC00", "#FF9900", "#FF6600", "#FF3300", "#FF0000", "#FF0095", 
    "#FC439F", "#FF88D3", "#FF99FF"
])
cape_norm = mcolors.BoundaryNorm(cape_bounds, cape_colors.N)

# ------------------------------
# DBZ-CMAX Farben
# ------------------------------
dbz_bounds = [0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 67, 70]
dbz_colors = ListedColormap([
    "#676767", "#FFFFFF", "#B3EFED", "#8CE7E2", "#00F5ED",
    "#00CEF0", "#01AFF4", "#028DF6", "#014FF7", "#0000F6",
    "#00FF01", "#01DF00", "#00D000", "#00BF00", "#00A701",
    "#019700", "#FFFF00", "#F9F000", "#EDD200", "#E7B500",
    "#FF5000", "#FF2801", "#F40000", "#EA0001", "#CC0000",
    "#FFC8FF", "#E9A1EA", "#D379D3", "#BE55BE", "#960E96"
])
dbz_norm = mcolors.BoundaryNorm(dbz_bounds, dbz_colors.N)

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

#-------------------------------
#Gesamtbewölkung-Farben
#------------------------------
# Farbskala für Gesamtbewölkung
cloud_bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # in cm
cloud_colors = ListedColormap([
    "#FFFF00", "#EEEE0B", "#DDDD17", "#CCCC22", "#BBBB2E",
    "#ABAB39", "#9A9A45", "#898950", "#78785C", "#676767"
])
cloud_norm = mcolors.BoundaryNorm(cloud_bounds, cloud_colors.N)

# ------------------------------
#Gesamtwassergehalt
# ------------------------------
twater_bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]  # in mm
twater_colors = ListedColormap([
        "#6E4A00", "#B49E62", "#D7CD13", "#B9F019", "#1ACF06",
        "#08534C", "#035DBE", "#2692FF", "#75BAFF", "#CBBFFF",
        "#EBA6FF", "#DD66FE", "#AC01DD", "#7C009E", "#673775",
        "#6B6B6B", "#818181", "#969696"
    ])

twater_norm = mcolors.BoundaryNorm(twater_bounds, twater_colors.N)

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
# Kartenparameter
# ------------------------------
FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT = FIG_W_PX / TOP_AREA_PX

# Bounding Box Deutschland (fix, keine GeoJSON nötig)
extent = [5, 16, 47, 56]

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
        data = ds["CAPE_ML"].values[0, :, :]
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
    # Figure
    # --------------------------
    scale = 0.9
    fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
    shift_up = 0.02
    ax = fig.add_axes([0.0, BOTTOM_AREA_PX / FIG_H_PX + shift_up, 1.0, TOP_AREA_PX / FIG_H_PX],
                      projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.set_axis_off()
    ax.set_aspect('auto')

    # Plot
    if var_type == "t2m":
        im = ax.pcolormesh(lon, lat, data, cmap=t2m_colors, norm=t2m_norm, shading="auto")

        contours = ax.contour(lon, lat, data, levels=t2m_bounds, colors='black', linewidths=0.3, alpha=0.6)

        # Anzahl der Werte, die angezeigt werden sollen
        n_labels = 40
        
        # 2D-Mesh für Maskierung
        lon2d, lat2d = np.meshgrid(lon, lat)
        
        # Alle gültigen Datenpunkte innerhalb des Extents
        lon_min, lon_max, lat_min, lat_max = extent
        valid_mask = np.isfinite(data) & (lon2d >= lon_min) & (lon2d <= lon_max) & (lat2d >= lat_min) & (lat2d <= lat_max)
        
        # Indizes der gültigen Punkte
        valid_indices = np.argwhere(valid_mask)

       # Abstand in Grad, innerhalb dessen keine Labels auf Städte gesetzt werden
        np.random.shuffle(valid_indices)
        min_city_dist = 1.0
        used_points = 0
        tried_points = set()
        texts = []
        while used_points < n_labels and len(tried_points) < len(valid_indices):
            i, j = valid_indices[np.random.randint(0, len(valid_indices))]
            if (i, j) in tried_points:
                continue
            tried_points.add((i, j))

            lon_pt, lat_pt = lon[j], lat[i]

            # Prüfen, ob zu nah an einer Stadt
            if any(np.hypot(lon_pt - city_lon, lat_pt - city_lat) < min_city_dist
                for city_lon, city_lat in zip(cities['lon'], cities['lat'])):
                # neuen Punkt versuchen – einfach continue
                continue
            
            val = data[i, j]
            txt = ax.text(lon_pt, lat_pt, f"{val:.0f}", fontsize=9,
                        ha='center', va='center', color='black', weight='bold')
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
            texts.append(txt)
            used_points += 1

        # Labels automatisch verschieben, um Überlappungen zu vermeiden
        adjust_text(texts, ax=ax, expand_text=(1.2, 1.2), arrowprops=dict(arrowstyle="-"))
        
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
    elif var_type == "cape_ml":
        im = ax.pcolormesh(lon, lat, data, cmap=cape_colors, norm=cape_norm, shading="auto")
    elif var_type == "dbz_cmax":
        im = ax.pcolormesh(lon, lat, data, cmap=dbz_colors, norm=dbz_norm, shading="auto")
    elif var_type == "wind":
        im = ax.pcolormesh(lon, lat, data, cmap=wind_colors, norm=wind_norm, shading="auto")
    elif var_type == "snow":
        im = ax.pcolormesh(lon2d, lat2d, data, cmap=snow_colors, norm=snow_norm, shading="auto")
    elif var_type == "cloud":
        im = ax.pcolormesh(lon, lat, data, cmap=cloud_colors, norm=cloud_norm, shading="auto")
    elif var_type == "twater":
        im = ax.pcolormesh(lon, lat, data, cmap=twater_colors, norm=twater_norm, shading="auto")
    elif var_type == "snowfall":
        im = ax.pcolormesh(lon, lat, data, cmap=snowfall_colors, norm=BoundaryNorm(snowfall_bounds, snowfall_colors.N), shading="auto")
    elif var_type == "pmsl":
    # Luftdruck-Daten
        im = ax.pcolormesh(lon, lat, data, cmap=pmsl_colors, norm=pmsl_norm, shading="auto")
        data_hpa = data  # data schon in hPa

        # Haupt-Isobaren (alle 4 hPa)
        main_levels = list(range(912, 1070, 4))
        # Feine Isobaren (alle 1 hPa)
        fine_levels = list(range(912, 1070, 1))

        main_levels = [lev for lev in main_levels if data_hpa.min() <= lev <= data_hpa.max()]
        fine_levels = [lev for lev in fine_levels if data_hpa.min() <= lev <= data_hpa.max()]

        # Feine Isobaren-Linien (transparent)
        cs_fine = ax.contour(lon, lat, data_hpa, levels=fine_levels,
                            colors='white', linewidths=0.3, alpha=0.8)

        # Haupt-Isobaren (dick, schwarz)
        cs_main = ax.contour(lon, lat, data_hpa, levels=main_levels,
                            colors='white', linewidths=1.2, alpha=1)

        used_points = set()  # global, für Haupt- und Feine-Isobaren
        texts = []  # Für adjust_text

        min_city_dist = 1.1  # Abstand in Grad zu Städten

        def place_random_labels(cs, n_labels):
            contour_points = []
            for level_segs in cs.allsegs:
                for seg in level_segs:
                    if seg.size > 0:
                        contour_points.extend(seg)
            contour_points = np.array(contour_points)

            # Punkte innerhalb des Extents filtern
            lon_min, lon_max, lat_min, lat_max = extent
            mask = (contour_points[:,0] >= lon_min) & (contour_points[:,0] <= lon_max) & \
                (contour_points[:,1] >= lat_min) & (contour_points[:,1] <= lat_max)
            contour_points = contour_points[mask]

            # Indexe auf Raster finden
            ij_points = [(np.abs(lat - lat_val).argmin(), np.abs(lon - lon_val).argmin()) 
                        for lon_val, lat_val in contour_points]

            # Duplikate entfernen
            ij_points = list(dict.fromkeys(ij_points))

            # Punkte filtern, die schon verwendet wurden oder zu nah an Städten liegen
            filtered_points = []
            for i,j in ij_points:
                lon_pt, lat_pt = lon[j], lat[i]

                # Abstand zu Städten prüfen
                if any(np.hypot(lon_pt - city_lon, lat_pt - city_lat) < min_city_dist
                    for city_lon, city_lat in zip(cities['lon'], cities['lat'])):
                    continue

                # Schon benutzt?
                if (i,j) in used_points:
                    continue

                filtered_points.append((i,j))

            # Zufällig n_labels auswählen
            if len(filtered_points) > n_labels:
                chosen_points = [filtered_points[i] for i in np.random.choice(len(filtered_points), n_labels, replace=False)]
            else:
                chosen_points = filtered_points

            # Werte auf die Karte setzen & als benutzt markieren
            for i, j in chosen_points:
                val = data_hpa[i, j]
                txt = ax.text(lon[j], lat[i], f"{val:.0f}", fontsize=10,
                            ha='center', va='center', color='black', weight='bold')
                txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
                texts.append(txt)
                used_points.add((i,j))

        # Zufällige Labels platzieren
        place_random_labels(cs_main, n_labels=5)  # Haupt-Isobaren
        place_random_labels(cs_fine, n_labels=5)  # Feine Isobaren

        # adjust_text aufrufen, um Überlappungen zwischen Labels zu vermeiden
        adjust_text(texts, ax=ax, expand_text=(1.2,1.2), arrowprops=dict(arrowstyle="-"))

    # Bundesländer-Grenzen aus Cartopy (statt GeoJSON)
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)
    for _, city in cities.iterrows():
        ax.plot(city["lon"], city["lat"], "o", markersize=6, markerfacecolor="black",
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        txt = ax.text(city["lon"]+0.1, city["lat"]+0.1, city["name"], fontsize=9,
                      color="black", weight="bold", zorder=6)
        txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_patch(mpatches.Rectangle((0,0),1,1, transform=ax.transAxes, fill=False, color="black", linewidth=2))

    # Legende
    legend_h_px = 50
    legend_bottom_px = 45
    if var_type in ["t2m","tp_acc","cape_ml","dbz_cmax","wind","snow", "cloud", "twater", "snowfall", "pmsl"]:
        bounds = t2m_bounds if var_type=="t2m" else tp_acc_bounds if var_type=="tp_acc" else cape_bounds if var_type=="cape_ml" else dbz_bounds if var_type=="dbz_cmax" else wind_bounds if var_type=="wind" else snow_bounds if var_type=="snow" else cloud_bounds if var_type=="cloud" else twater_bounds if var_type=="twater" else snowfall_bounds if var_type=="snowfall" else pmsl_bounds_colors 
        cbar_ax = fig.add_axes([0.03, legend_bottom_px / FIG_H_PX, 0.94, legend_h_px / FIG_H_PX])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=bounds)
        cbar.ax.tick_params(colors="black", labelsize=7)
        cbar.outline.set_edgecolor("black")
        cbar.ax.set_facecolor("white")

        # Für pmsl nur jeden 10. hPa Tick beschriften
        if var_type=="pmsl":
            tick_labels = [str(tick) if tick % 8 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type == "t2m":
            tick_labels = [str(tick) if tick % 4 == 0 else "" for tick in bounds]
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
        "tp_acc": "Akkumulierter Niederschlag (mm)",
        "cape_ml": "CAPE-Index (J/kg)",
        "dbz_cmax": "Sim. max. Radarreflektivität (dBZ)",
        "cloud": "Gesamtbewölkung (%)",
        "wind": "Windböen (km/h)",
        "snow": "Schneehöhe (cm)",
        "twater": "Gesamtwassergehalt (mm)",
        "snowfall": "Schneefallgrenze (m)",
        "pmsl": "Luftdruck auf Meereshöhe (hPa)"
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
