import requests
from bs4 import BeautifulSoup as bts
import pandas as pd
import re
import numpy as np
import time

link = 'https://sofifa.com/players?type=all&lg%5B0%5D=13&lg%5B1%5D=16&lg%5B2%5D=19&lg%5B3%5D=31&lg%5B4%5D=53&col=vl&sort=desc&showCol%5B%5D=pi&showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=wi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=pt&showCol%5B%5D=bo&showCol%5B%5D=bp&showCol%5B%5D=gu&showCol%5B%5D=vl&showCol%5B%5D=wg&showCol%5B%5D=rc&showCol%5B%5D=ta&showCol%5B%5D=cr&showCol%5B%5D=fi&showCol%5B%5D=he&showCol%5B%5D=sh&showCol%5B%5D=vo&showCol%5B%5D=ts&showCol%5B%5D=dr&showCol%5B%5D=cu&showCol%5B%5D=fr&showCol%5B%5D=lo&showCol%5B%5D=bl&showCol%5B%5D=to&showCol%5B%5D=ac&showCol%5B%5D=sp&showCol%5B%5D=ag&showCol%5B%5D=re&showCol%5B%5D=ba&showCol%5B%5D=tp&showCol%5B%5D=so&showCol%5B%5D=ju&showCol%5B%5D=st&showCol%5B%5D=sr&showCol%5B%5D=ln&showCol%5B%5D=te&showCol%5B%5D=ar&showCol%5B%5D=in&showCol%5B%5D=po&showCol%5B%5D=vi&showCol%5B%5D=pe&showCol%5B%5D=cm&showCol%5B%5D=td&showCol%5B%5D=ma&showCol%5B%5D=sa&showCol%5B%5D=sl&showCol%5B%5D=tg&showCol%5B%5D=gd&showCol%5B%5D=gh&showCol%5B%5D=gc&showCol%5B%5D=gp&showCol%5B%5D=gr&showCol%5B%5D=tt&showCol%5B%5D=bs&showCol%5B%5D=ir&showCol%5B%5D=pac&showCol%5B%5D=sho&showCol%5B%5D=pas&showCol%5B%5D=dri&showCol%5B%5D=def&showCol%5B%5D=phy'

def getAndParseURL(url):
    result = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}) # Safari/537.36. Chrome/103.0.0.0
    soup = bts(result.text, "html.parser")
    return soup
pages = [link]
for page in range(60,2880,60):
    pages.append(link+"&offset="+str(page))

players = []
i = 0 
for page in pages:
    html = getAndParseURL(page)
    table = html.find("table")
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        player = {"name": cols[1].get_text().strip()} 
        for col in cols[2:]: 
            header = table.find_all("th")[cols.index(col)].get_text().strip()
            player[header] = col.get_text().strip() 
        players.append(player)
    print(i)
    i+=1
    time.sleep(1)

df = pd.DataFrame(players)
df.to_csv('top5_leagues_all_players.csv', index=False)


