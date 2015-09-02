scp achandrasekhar@doritos.snl.salk.edu:/home/achandrasekhar/Documents/ant_deviate.csv ant_deviate2.csv
python clean_csv.py ant_deviate2.csv ant_deviate3.csv
rm -f ant_deviate2.csv
python plot_ant_deviate.py ant_deviate3.csv deviate
mv ant_deviate3.csv ant_deviate.csv
