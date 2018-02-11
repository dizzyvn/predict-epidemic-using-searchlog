from __future__ import absolute_import, division, print_function
from functools  import reduce

import sys

import gft
import proposed
import common.printer    as printer
import common.visualizer as visualizer

disease_names = ['Influenza (The flu)', 'Hand, foot and mouth disease (HFMD)', 'Chickenpox (Varicella)',
                 'Erythema infectiosum (Fifth disease)','Pharyngoconjunctival fever (PCF)',
                 'Herpangina (Mouth blisters)', 'Streptococcal pharyngitis (Strep throat)',
                 'Epidemic keratoconjunctivitis (EKC)', 'Gastroenteritis (Infection diarrhea)',
                 'Mycoplasma pneumonia (Walking pneumonia)', ]

if __name__ == '__main__':
    config = list(filter(lambda x: x[0] is '-', sys.argv[1:]))
    config_kwargs = reduce(lambda x, y: {y.split('=')[0][1:]: y.split('=')[1], **x}, config, {})

    # GET TYPE OF DISEASE TO EXPERIMENT
    if 'disease' not in config_kwargs or config_kwargs['disease'] == 'all':
        diseases = range(len(disease_names))
    else:
        diseases = [int(config_kwargs['disease'])]

    # GET THE TIME-LAG PARAMETER
    if 'lag' in config_kwargs:
        lag = int(config_kwargs['lag'])
    else:
        lag = 0

    # GET THE PLOT PARAMETER
    if 'plot' not in config_kwargs or config_kwargs['disease'] == 'false':
        plot = False
    else:
        plot = True

    gft_mapes = []
    gft_coefs = []
    pp_mapes  = []
    pp_coefs  = []

    for disease_no in diseases:
        print('\n\n===========================================================')
        gft_mape = 182.830810804
        gft_coef = 0.596907379564
        pp_mape = 82.830810804
        pp_coef = 0.596907379564
        print('Conducting experiment on', disease_names[disease_no], '\nwith time-lag =', lag)
        print('\n>>> PROPOSED method')
        pp_mape, pp_coef, pp_prediction = proposed.experiment(disease_no, lag)

        print('\n>>> GFT method')
        gft_mape, gft_coef, gft_prediction = gft.experiment(disease_no, lag)

        print('---- RESULTS ----------')
        print('GFT      : Mape value = {:7.2f}, Coef = {:7.2f}'.format(gft_mape, gft_coef))
        print('Proposed : Mape value = {:7.2f}, Coef = {:7.2f}'.format(pp_mape , pp_coef ))

        gft_mapes.append(gft_mape)
        gft_coefs.append(gft_coef)
        pp_mapes.append(pp_mape)
        pp_coefs.append(pp_coef)

        if plot == True:
            visualizer.plot_results(pp_prediction, gft_prediction,
                                    disease_names[disease_no], disease_no, lag)

    printer.display_result_table(gft_mapes, gft_coefs, pp_mapes, pp_coefs, diseases)
