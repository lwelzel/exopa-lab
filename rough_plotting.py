import mesaPlot as mp
from nugridpy import mesa as ms
from pathlib import Path

m=mp.MESA()
m.log_fold = './LOGS/run-m_core-3_0-f-0_1'
p=mp.plot()


if __name__ == "__main__":
    # s = ms.history_data('./LOGS/run-m_core-3_0-f-0_1')
    # s.hrd()

    m.loadHistory()
    print(m.hist.data)

    p.plotHistory(m, xaxis='log_star_age', y1='log_center_T', y2='he_core_mass')
    p.plotHR(m)