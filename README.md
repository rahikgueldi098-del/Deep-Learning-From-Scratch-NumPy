# Réseau de Neurones Multicouche from Scratch (NumPy)

Implémentation complète d'un réseau de neurones artificiels (Deep Learning) codé intégralement en Python avec **NumPy**, sans utiliser de frameworks de haut niveau (PyTorch, TensorFlow).

Ce projet illustre le fonctionnement interne du Deep Learning : de la propagation avant au calcul vectoriel des gradients par rétropropagation.

## Architecture & Mathématiques

L'algorithme repose sur les étapes fondamentales suivantes :

1. **Forward Propagation (Propagation avant) :**
   $$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
   $$A^{[l]} = \sigma(Z^{[l]})$$
   *Fonction d'activation Sigmoïde :* $\sigma(z) = \frac{1}{1 + e^{-z}}$

2. **Backward Propagation (Rétropropagation du gradient) :**
   Calcul vectorisé des dérivées partielles $\frac{\partial L}{\partial W}$ et $\frac{\partial L}{\partial b}$ sur $L$ couches.

3. **Gradient Descent (Descente de gradient) :**
   Mise à jour dynamique des poids et des biais avec un taux d'apprentissage ($\alpha$) :
   $$W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]}$$
   $$b^{[l]} = b^{[l]} - \alpha \cdot db^{[l]}$$

## Fonctionnalités
- **Architecture Modulaire L-Couches :** Support d'un nombre flexible de couches cachées.
- **Calcul Vectorisé :** Traitement efficace des matrices via NumPy.
- **Algorithme d'Apprentissage :** Implémentation manuelle de la Rétropropagation du gradient et du Gradient Descent.

## Exécution
Pour exécuter le test de validation du réseau :
```bash
python dl.py