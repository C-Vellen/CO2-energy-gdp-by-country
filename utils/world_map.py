import json
from utils.display import units, couleurs_vives


def html_world_map(df, features):
    """create world map html dans un fichier html"""
    with open("../data/world_map.json") as f:
        wm = json.load(f)

    html_map = '<!DOCTYPE html><html lang="fr"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><style type="text/css">body{font-family:helvetica;}</style></head><body><div id="container">'
    html_map += f'<h1 style="text-align:center; font-size:24px;">Regroupements des pays suivant les variables :</h1><h1 style="text-align:center; font-size:20px;">{", ".join([f.replace('_per_', '/') for f in features])}. Année {str(list(set(df['year']))[0])}.</h1>'
    html_map += '<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 -50 2000 1000" xml:space="preserve">'

    legend = {f: f"{f.replace('_per_', '/')} [{units[f]}]" for f in features}

    error_code = []
    # continent:
    for continent_code, continent_map in wm.items():
        html_map += f"<g class='{continent_code}'>\n"

        # pays:
        for country_code, country_map in continent_map.items():
            try:
                html_map += (
                    f"<path data-ISOCODE={country_code} data-country='{df[df["iso_code"]==country_code[:3]]["country"].iloc[0]}'"
                    + " ".join(
                        [
                            f"data-{feat}={df[df["iso_code"]==country_code[:3]][feat].iloc[0]:.2f}"
                            for feat in features
                        ]
                    )
                    + f" class='country {country_code[:3]}' d='{country_map}' fill={df[df['iso_code']==country_code[:3]]['cluster_color'].iloc[0]} stroke={'#555'} opacity=0.8></path>"
                )
            except IndexError:
                html_map += f"<path data-ISOCODE={country_code} data-info='no data' class='country {country_code[:3]}' d='{country_map}' fill={'#aaa'} stroke={'#555'} opacity=0.8></path>"
                error_code.append(country_code[:3])

        html_map += "</g>"

    # légende :
    html_map += "<g style='transform:translate(150px, 700px);'>"
    for i in range(len(set(df["cluster"]))):
        html_map += f"<g style='transform:translate(0,{i*30}px);'> <rect x='0' y='0' width='60' height='20' fill='{couleurs_vives[i]}' stroke='#000'></rect><text x='120' y='20' style='font-size:20px;'>cluster: {i}</text> </g>"
    html_map += f"<g style='transform:translate(0,{len(set(df["cluster"]))*30}px);'> <rect x='0' y='0' width='60' height='20' fill='#aaa' stroke='#000'></rect><text x='120' y='20' style='font-size:18px;'>Manque de données</text> </g>"
    html_map += (
        "</g></svg></div></body><script src='outputs/tooltip.js'></script></html>"
    )

    list_error = [code for code in set(error_code)]
    list_error.sort()
    print("Error codes: ", len(list_error), "manque de données")
    print(", ".join(list_error))
    with open("../index.html", "w") as f:
        f.write(html_map)
