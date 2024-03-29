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
agns = ["bll", "fsrq", "bcu", "agn", 'FSRQ', 'BLL', 'BCU', "agn"]
psrs = ['psr']
others = ['spp', 'bin', 'rdg', 'RDG', 'nlsy1', 'NLSY1', 'ssrq', 'sey']
unids = ['']

coord = SkyCoord(table["GLON"], table["GLAT"], frame="galactic")  # type: SkyCoord

for unid in unids:
    index = np.array([_.strip().lower() == unid for _ in table["CLASS1"]])
    label = unid if unid else "unid"
    plot2 = ax.scatter(coord[index].l.wrap_at(180 * u.degree).radian, coord[index].b.radian, label=label, s=250,
                       c='#707C74', marker='x')
# #91AD70

ax.grid(True)
plt.legend([plot2], ["Unid"], loc=1, title="Class", prop={'size': 18}, title_fontsize=18)

plt.savefig("/Users/kawakami/Desktop/fight/Distribution/data/Plot_Unid.png", format='png', dpi=100)
