L’objectif de ce projet est d’implémenter une méthode de comparaison d’images
en se basant sur l’histogramme couleur de celles-ci, afin d’en étudier l’efficacité
sur différents ensembles d’images dans différents espaces de couleurs. Le papier ”Swain et Ballard, Color indexing, International Jour-
nal of ComputerVision,7 :1, 11-32 (1991)” contient l'espace de couleur utilisé dans ce projet
ainsi que la formule d’intersection d’histogrammes utilisée pour la comparaison d'images.

- rg = r − g
- by = 2 ∗ b − r − g
- wb = r + g + b


