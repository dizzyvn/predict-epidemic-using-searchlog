# Seasonal-adjustment based feature selection method for predicting epidemic with large-scale search engine logs [KDD2019]
A seasonal adjustment based feature selection and prediction modeling for predicting epidemic with search engine logs. This repository contains the source code and a part of the data to reprocedure the experiment. Since we're not able to public all 2.5 milion search terms, we instead publish the data of the search terms that actually selected when we conducted the experiment with 2.5 milion search terms and a number of search terms with high score. 

More information in our paper.
https://dl.acm.org/doi/10.1145/3292500.3330766

How to use
======
For specific disease
```python
python main.py --disease 0 --lag 0 --plot true
python main.py --disease 2 --lag 4 --plot false
```

Run all diseases at a time
```python
python main.py --lag 0
python main.py --lag 3
```

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-right">No</th>
<th scope="col" class="org-left">Disease name</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-right">0</td>
<td class="org-left">Influenza (The flu)</td>
</tr>


<tr>
<td class="org-right">1</td>
<td class="org-left">Hand, foot and mouth disease (HFMD)</td>
</tr>


<tr>
<td class="org-right">2</td>
<td class="org-left">Chickenpox (Varicella)</td>
</tr>


<tr>
<td class="org-right">3</td>
<td class="org-left">Erythema infectiosum (Fifth disease)</td>
</tr>


<tr>
<td class="org-right">4</td>
<td class="org-left">Pharyngoconjunctival fever (PCF)</td>
</tr>


<tr>
<td class="org-right">5</td>
<td class="org-left">Herpangina (Mouth blisters)</td>
</tr>


<tr>
<td class="org-right">6</td>
<td class="org-left">Streptococcal pharyngitis (Strep throat)</td>
</tr>


<tr>
<td class="org-right">7</td>
<td class="org-left">Epidemic keratoconjunctivitis (EKC)</td>
</tr>


<tr>
<td class="org-right">8</td>
<td class="org-left">Gastroenteritis (Infectious diarrhea)</td>
</tr>


<tr>
<td class="org-right">9</td>
<td class="org-left">Mycoplasma pneumonia (Walking pneumonia)</td>
</tr>
</tbody>
</table>
