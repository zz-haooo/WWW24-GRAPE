echo "###############################  Running on Cora  ###############################################"
python main.py --dataset=cora
echo "###############################  Running on CiteSeer  ###############################################"
python main.py --dataset=citeseer
echo "###############################  Running on PubMed  ###############################################"
python main.py --dataset=pubmed
echo "###############################  Running on Amazon Photo  ###############################################"
python main.py --dataset=ama_photo
echo "###############################  Running on Amazon Computers  ###############################################"
python main.py --dataset=ama_computer
echo "###############################  Running on Wiki CS  ###############################################"
python main.py --dataset=wiki
echo "###############################  Running on Coauthor CS  ###############################################"
python main.py --dataset=co_cs
echo "###############################  Running on Coauthor Physics  ###############################################"
python main.py --dataset=co_physics