rm -rf holidays
wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz
tar xzf jpg1.tar.gz
wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz
tar xzf jpg2.tar.gz
mv jpg holidays
rm jpg*

# Getting Ground Truth file
wget http://lear.inrialpes.fr/people/jegou/code/eval_holidays.tgz
tar xzf eval_holidays.tgz
mv eval_holidays/holidays_images.dat holidays
rm -rf eval_holidays*

rm eval_holidays.py
wget https://raw.githubusercontent.com/emiliofidalgo/11762_IRIC/master/eval/eval_holidays.py

rm -rf siftgeo
wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/siftgeo.tar.gz
tar xzf siftgeo.tar.gz
rm *.gz

# Downloading the index tools
rm index_utils.py
wget https://raw.githubusercontent.com/emiliofidalgo/11762_IRIC/master/index/index_utils.py

# Downloading visual dictionaries
rm -rf clust
wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/clust.tar.gz
tar xzf clust.tar.gz
rm *.gz