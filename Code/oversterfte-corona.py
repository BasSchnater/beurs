##### OVERSTERFTE CORONA #####
import pandas as pd
import seaborn as sns
import cbsodata

sterfte = pd.DataFrame(cbsodata.get_data('70895ned'))
sterfte = sterfte[['RegioS','Perioden']]

gooi = sterfte[sterfte['RegioS'] == 'Het Gooi en Vechtstreek (CR)']
gooi_pivot = gooi.pivot_table(index='Perioden', values='TotaalMannenEnVrouwen_9')
gooi_pivot.plot.bar()
plt.ylim(0,5000)
plt.show()

huizen = huizen.rename(columns={'GemiddeldeVerkoopprijs_1':'prijs_gem','RegioS':'Regio'}).drop(columns=['ID'])
huizen_ams = huizen[huizen['Regio'] == 'Amsterdam']
huizen_ams.index = huizen_ams['Perioden']
huizen_ams['pct_change'] = huizen_ams['prijs_gem'].pct_change()
huizen_ams['prijs_gem'].plot(label='Amsterdam')
huizen_lelystad = huizen[huizen['Regio'] == 'Lelystad']
huizen_lelystad.index = huizen_lelystad['Perioden']
huizen_lelystad['pct_change'] = huizen_lelystad['prijs_gem'].pct_change()
huizen_lelystad['prijs_gem'].plot(label='Lelystad')
plt.title('Groei gem. huizenprijs')
plt.ylabel('€ gem. huizenprijs')
plt.legend()
plt.show()
