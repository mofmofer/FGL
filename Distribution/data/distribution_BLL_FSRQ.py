"""
AGN、PSR、Unidの全天マップを出力します.
"""
# coding=utf-8

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

# 4FGLソースリストの読み込み
filename = '/Users/kawakami/Desktop/fight/gll_psc_v19.fit'
table = Table.read(filename, hdu=1)

extended_source_table = Table.read(filename, hdu="ExtendedSources")
scalar_column_names = tuple(name for name in table.colnames if len(table[name].shape) <= 1)

# 表示する画像サイズ
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection="aitoff")

#  ソースクラスのリスト
source_classes = ['psr', 'spp', 'bin', 'BCU', 'bcu', 'BLL', 'bll', 'FSRQ', 'fsrq', 'rdg', 'RDG', 'nlsy1', 'NLSY1',
                  'agn', 'ssrq', 'sey']
blls = ["bll", "BLL"]
fsrqs = ['fsrq','FSRQ']
others = ["bcu", "agn", 'BCU', "agn"]
unids = ['']

coord = SkyCoord(table["GLON"], table["GLAT"], frame="galactic")  # type: SkyCoord

for agn in blls:
    index = np.array([_.strip().lower() == agn for _ in table["CLASS1"]])
    label = agn if agn else "unid"
    plot1 = ax.scatter(coord[index].l.wrap_at(180 * u.degree).radian, coord[index].b.radian, label=label, s=50,
                       c='#D2D7D3', linewidths="1", edgecolors="#678CB7", marker='+')


"""
for unid in unids:
index = np.array([_.strip().lower() == unid for _ in table["CLASS1"]])
label = unid if unid else "unid"
plot2 = ax.scatter(coord[index].l.wrap_at(180 * u.degree).radian, coord[index].b.radian, label=label, s=50,
c='#91AD70', marker='x')
"""
for psr in fsrqs:
    index = np.array([_.strip().lower() == psr for _ in table["CLASS1"]])
    label = psr if psr else "unid"
    plot3 = ax.scatter(coord[index].l.wrap_at(180 * u.degree).radian, coord[index].b.radian, label=label, s=50,
                       c='#1C8A0B', marker='^')
"""
for other in others:
index = np.array([_.strip().lower() == other for _ in table["CLASS1"]])
label = other if other else "unid"
plot4 = ax.scatter(coord[index].l.wrap_at(180 * u.degree).radian, coord[index].b.radian, label=label, s=50,
c='#000000', marker='x')
"""

ax.grid(True)
plt.legend([plot1, plot3],
           ["BLL", "FSRQ"],
           loc=1,
           title="Class", prop={'size': 18}, title_fontsize=18)

plt.savefig("/Users/kawakami/Desktop/fight/Distribution/data/Plot_AGNs.png", format='png', dpi=100)
