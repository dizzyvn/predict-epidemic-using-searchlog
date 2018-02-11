disease_names = ['Influenza (The flu)', 'Hand, foot and mouth disease (HFMD)', 'Chickenpox (Varicella)',
                 'Erythema infectiosum (Fifth disease)','Pharyngoconjunctival fever (PCF)',
                 'Herpangina (Mouth blisters)', 'Streptococcal pharyngitis (Strep throat)',
                 'Epidemic keratoconjunctivitis (EKC)', 'Gastroenteritis (Infection diarrhea)',
                 'Mycoplasma pneumonia (Walking pneumonia)', ]


def display_top_terms(term_list, name):
    print('- Top-5 terms ', name)
    for term, score in term_list[:5]:
        print('{:5.2f} | '.format(score) + term)


def display_result_table(gft_mapes, gft_coefs, pp_mapes, pp_coefs, diseases):
    print('\n-----------------------------------------')
    print('  GFT method   |Proposed method| DISEASE')
    print('---------------+---------------+---------')
    print('  MAPE | COEF  |  MAPE |   COEF| ')
    for i, disease_no in enumerate(diseases):
        print('{:7.2f}|{:7.2f}|{:7.2f}|{:7.2f}|'
              .format(gft_mapes[i], gft_coefs[i], pp_mapes[i], pp_coefs[i])
              + disease_names[disease_no])