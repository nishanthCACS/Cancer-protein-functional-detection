# -*- coding: utf-8 -*-
"""
Created on %%(03-Sep-2018) 15.30Pm

@author: %A.Nishanth C00294860
"""
import os
from class_probabilty_weigtht_cal_results_from_NN  import probability_weight_cal_results_from_NN

minimum_prob_decide = 0.25

#%% For Tier_1

working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping"
working_dir_NN_prob = "".join([working_dir_part, "/Tier_1_NN_source"])

"BLOCK 1 out of 7"
name = "ONGO"
working_dir_fraction = "".join([working_dir_part, "/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_1_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
ONGO = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
ONGO.documentation_prob()

#% creating the model for TSG
"BLOCK 2 out of 7"
name = "TSG"
working_dir_fraction = "".join([working_dir_part, "/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_1_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
TSG = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
TSG.documentation_prob()

#% creating the model for Fusion
"BLOCK 3 out of 7"
name = "Fusion"
working_dir_fraction = "".join([working_dir_part, "/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_1_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
Fusion = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
Fusion.documentation_prob()

"BLOCK 4 out of 7"
#% creating the model for TSG_Fusion
name = "TSG_Fusion"
working_dir_fraction = "".join([working_dir_part, "/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_1_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
TSG_Fusion = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
TSG_Fusion.documentation_prob()

"BLOCK 5 out of 7"
#% creating the model for ONGO_Fusion
name = "ONGO_Fusion"
working_dir_fraction = "".join([working_dir_part, "/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_1_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
ONGO_Fusion = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
ONGO_Fusion.documentation_prob()

"BLOCK 6 out of 7"
#% creating the model for ONGO_TSG
name = "ONGO_TSG"
working_dir_fraction = "".join([working_dir_part, "/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_1_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
ONGO_TSG = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
ONGO_TSG.documentation_prob()

"BLOCK 7 out of 7"
#% creating the model for ONGO_TSG_Fusion
name = "ONGO_TSG_Fusion"
working_dir_fraction = "".join([working_dir_part, "/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_1_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
ONGO_TSG_Fusion = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
ONGO_TSG_Fusion.documentation_prob()
#%% Tier_2
"""           For Tier_2    """
working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping"
working_dir_NN_prob = "".join([working_dir_part, "/Tier_2_NN_source"])

"BLOCK 1 out of 6"
name = "ONGO"
working_dir_fraction = "".join([working_dir_part, "/Tier_2_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_2_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
ONGO = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
ONGO.documentation_prob()

#% creating the model for TSG
"BLOCK 2 out of 6"
name = "TSG"
working_dir_fraction = "".join([working_dir_part, "/Tier_2_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_2_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
TSG = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
TSG.documentation_prob()

#% creating the model for Fusion
"BLOCK 3 out of 6"
name = "Fusion"
working_dir_fraction = "".join([working_dir_part, "/Tier_2_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_2_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
Fusion = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
Fusion.documentation_prob()

"BLOCK 4 out of 6"
#% creating the model for ONGO_Fusion
name = "ONGO_Fusion"
working_dir_fraction = "".join([working_dir_part, "/Tier_2_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_2_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
ONGO_Fusion = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
ONGO_Fusion.documentation_prob()

"BLOCK 5 out of 6"
#% creating the model for ONGO_TSG
name = "ONGO_TSG"
working_dir_fraction = "".join([working_dir_part, "/Tier_2_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_2_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
ONGO_TSG = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
ONGO_TSG.documentation_prob()

"BLOCK 6 out of 6"
#% creating the model for ONGO_TSG_Fusion
name = "Notannotated"
working_dir_fraction = "".join([working_dir_part, "/Tier_2_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_2_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
Notannotated = probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, saving_dir,name,minimum_prob_decide)
Notannotated.documentation_prob()
