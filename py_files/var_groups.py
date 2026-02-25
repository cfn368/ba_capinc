import pandas as pd

# ========== ========== ========== ========== ==========
# 1. simulation variables
model_var = {
    'K':        r'$K$',
    'q':        r'$q$',
    'pI':       r'$p_I$',
    'I':        r'$I$',
    'C':        r'$C$',
    'rC_gross': r'$r_K$',
    'sK':       r'$s_K$',
    's1':       r'$s_{L}$',
    's2':       r'$s_{2L}$',
    'L1C':      r'$L_{C}$',
    'L2C':      r'$L_{2C}$',
    'L1I':      r'$L_{I}$',
    'L2I':      r'$L_{2I}$',
    'tau':      r'$\tau$',
    'Y':        r'$Y$',
    'w1C':      r'$w_{C}$',
    'w2C':      r'$w_{2C}$',
    'w1I':      r'$w_{I}$',
    'w2I':      r'$w_{2I}$',
    
    'wC':       r'$w_{C}$',
    'wI':       r'$w_{I}$',
    'sL':       r'$s_{L}$',
    'LC':       r'$L_{C}$',
    'LI':       r'$L_{I}$',
}

def panels(gamma): 
    return [
    # ("Y", r"$Y$"),
    # ("C", r"$C$"),
    # ('pII', r"$p_II$"),
    # ("wC", r"$w_C$"),
    # ("wI", r"$w_I$"),
    ("LS_C", r"$LS_C$"),
    ("LS_I", r"$LS_I$"),
    ("LS", r"$LS$"),
    ("LS_gamma", rf"Adj. $LS,\; (\gamma={gamma})$"),
]
    
# ========== ========== ========== ========== ==========
# 2. investment share

# main categories
tilgange2_var = [
    'TA', 'TB', 'TCA', 'TCB', 'TCC', 'TCD', 'TCE', 'TCF', 'TCG', 'TCH', 'TCI',
    'TCJ', 'TCK', 'TCL', 'TCM', 'TD', 'TE', 'TF', 'TG', 'TH', 'TI', 'TJA',
    'TJB', 'TJC', 'TK', 'TLA', 'TLB', 'TMA', 'TMB', 'TMC', 'TN', 'TO', 'TP',
    'TQA', 'TQB', 'TR', 'TSA', 'TSB'    
]

anveldelse_var = [
    'AA', 'AB', 'ACA', 'ACB', 'ACC', 'ACD', 'ACE', 'ACF', 'ACG', 'ACH', 'ACI',
    'ACJ', 'ACK', 'ACL', 'ACM', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJA',
    'AJB', 'AJC', 'AK', 'ALA', 'ALB', 'AMA', 'AMB', 'AMC', 'AN', 'AO', 'AP',
    'AQA', 'AQB', 'AR', 'ASA', 'ASB'    
]

# link between sub and main categories
sub_to_parent = {
    "01000": "A",  "02000": "A",  "03000": "A",  "06090": "B",  "10120": "CA",
    "13150": "CB", "16000": "CC", "17000": "CC", "18000": "CC", "19000": "CD",
    "20000": "CE", "21000": "CF", "22000": "CG", "23000": "CG", "24000": "CH",
    "25000": "CH", "26000": "CI", "27000": "CJ", "28000": "CK", "29000": "CL",
    "30000": "CL", "31320": "CM", "33000": "CM", "35000": "D",  "36000": "E",
    "37390": "E",  "41430": "F",  "45000": "G",  "46000": "G",  "47000": "G",
    "49000": "H",  "50000": "H",  "51000": "H",  "52000": "H",  "53000": "H",
    "55560": "I",  "58000": "JA", "59600": "JA", "61000": "JB", "62630": "JC",
    "64000": "K",  "65000": "K",  "66000": "K",  "68100": "LA", "68300": "LA",
    "68203": "LB", "68204": "LB", "69700": "MA", "71000": "MA", "72001": "MB",
    "72002": "MB", "73000": "MC", "74750": "MC", "77000": "N",  "78000": "N",
    "79000": "N",  "80820": "N",  "84202": "O",  "84101": "O",  "85202": "P",
    "85101": "P",  "86000": "QA", "87880": "QB", "90920": "R",  "93000": "R",
    "94000": "SA", "95000": "SA", "96000": "SA", "97000": "SB",
}

# tangible or intangible
industry_classification = {
    # Agriculture, forestry, fishing - tangible (physical goods)
    '01000': 'tangible',
    '02000': 'tangible',
    '03000': 'tangible',
    
    # Mining
    '06090': 'tangible',
    
    # Manufacturing - all tangible (physical goods production)
    '10120': 'tangible',
    '13150': 'tangible',
    '16000': 'tangible',
    '17000': 'tangible',
    '18000': 'tangible',
    '19000': 'tangible',
    '20000': 'tangible',
    '21000': 'intangible', # pharma
    '22000': 'tangible',
    '23000': 'tangible',
    '24000': 'tangible',
    '25000': 'tangible',
    '26000': 'intangible', # software
    '27000': 'tangible',
    '28000': 'tangible',
    '29000': 'tangible',
    '30000': 'tangible',
    '31320': 'tangible',
    '33000': 'tangible',
    
    # Utilities - tangible
    '35000': 'tangible',
    '36000': 'tangible',
    '37390': 'tangible',
    
    # Construction - tangible (structures)
    '41430': 'tangible',
    
    # Wholesale and retail trade - tangible (goods distribution)
    '45000': 'tangible',
    '46000': 'tangible',
    '47000': 'tangible',
    
    # Transportation - tangible
    '49000': 'tangible',
    '50000': 'tangible',
    '51000': 'tangible',
    '52000': 'tangible',
    '53000': 'tangible',
    
    # Accommodation and food service - tangible
    '55560': 'tangible',
    
    # Publishing, TV, radio - intangible (intellectual property)
    '58000': 'intangible',
    '59600': 'intangible',
    
    # Telecommunications - intangible
    '61000': 'intangible',
    
    # IT and information services - intangible
    '62630': 'intangible',
    
    # Financial and insurance - intangible
    '64000': 'intangible',
    '65000': 'intangible',
    '66000': 'intangible',
    
    # Real estate - tangible (structures/dwellings)
    '68100': 'tangible',
    '68300': 'tangible',
    '68203': 'tangible',
    '68204': 'tangible',
    
    # Knowledge-based / professional services - intangible (organizational capital)
    '69700': 'organisational',  # Legal, accounting, management consultancy
    '71000': 'organisational',  # Architectural and engineering activities
    
    # R&D - intangible
    '72001': 'intangible',
    '72002': 'intangible',
    
    # Advertising and other business services - intangible (organizational capital)
    '73000': 'organisational',
    '74750': 'organisational',
    
    # Administrative/operational services - intangible (organizational capital)
    '77000': 'tangible',        # Rental and leasing (of physical assets)
    '78000': 'organisational',  # Employment activities
    '79000': 'intangible',      # Travel agent activities
    '80820': 'organisational',  # Security, building services, other business support
    
    # Public administration - intangible
    '84202': 'intangible',
    '84101': 'intangible',
    
    # Education - intangible
    '85202': 'intangible',
    '85101': 'intangible',
    
    # Health and social work - intangible
    '86000': 'intangible',
    '87880': 'intangible',
    
    # Arts, entertainment, recreation - intangible
    '90920': 'intangible',
    '93000': 'intangible',
    
    # Other service activities - intangible
    '94000': 'intangible',
    '95000': 'tangible',   # Repair of personal goods (physical)
    '96000': 'intangible',
    
    # Household employers
    '97000': 'intangible',
}

# Map industries to investment TYPE (not tangible/intangible)

investment_type = {
    # Structures
    '41430': 'structures',
    '68100': 'structures',
    '68300': 'structures',
    '68203': 'structures',
    '68204': 'structures',

    # Equipment
    '01000': 'equipment',  '02000': 'equipment',  '03000': 'equipment',
    '06090': 'equipment',
    '10120': 'equipment',  '13150': 'equipment',  '16000': 'equipment',
    '17000': 'equipment',  '18000': 'equipment',  '19000': 'equipment',
    '20000': 'equipment',  '22000': 'equipment',  '23000': 'equipment',
    '24000': 'equipment',  '25000': 'equipment',  '26000': 'equipment',
    '27000': 'equipment',  '28000': 'equipment',
    '29000': 'equipment',  '30000': 'equipment',  '31320': 'equipment',
    '33000': 'equipment',
    '35000': 'equipment',  '36000': 'equipment',  '37390': 'equipment',
    '45000': 'equipment',  '46000': 'equipment',  '47000': 'equipment',
    '49000': 'equipment',  '50000': 'equipment',  '51000': 'equipment',
    '52000': 'equipment',  '53000': 'equipment',
    '55560': 'equipment',
    '77000': 'equipment',
    '95000': 'equipment',

    # Intellectual property
    '21000': 'intellectual_property',  # Pharma (R&D-driven)
    '58000': 'intellectual_property',
    '59600': 'intellectual_property',
    '61000': 'intellectual_property',
    '62630': 'intellectual_property',
    '64000': 'intellectual_property',  '65000': 'intellectual_property',
    '66000': 'intellectual_property',
    '72001': 'intellectual_property',  '72002': 'intellectual_property',
    '79000': 'intellectual_property',
    '84202': 'intellectual_property',  '84101': 'intellectual_property',
    '85202': 'intellectual_property',  '85101': 'intellectual_property',
    '86000': 'intellectual_property',  '87880': 'intellectual_property',
    '90920': 'intellectual_property',  '93000': 'intellectual_property',
    '94000': 'intellectual_property',  '96000': 'intellectual_property',
    '97000': 'intellectual_property',

    # Organizational services
    '69700': 'organizational',
    '71000': 'organizational',
    '73000': 'organizational',
    '74750': 'organizational',
    '78000': 'organizational',
    '80820': 'organizational',
}


# ========== ========== ========== ========== ==========
# 3. labour share

mothers = pd.Index([
    'A Agriculture, forestry and fishing', 'B Mining and quarrying',
    'C Manufacturing',
    'D Electricity, gas, steam and air conditioning supply',
    'E Water supply, sewerage and waste management', 'F Construction',
    'G Wholesale and retail trade', 'H Transportation',
    'I Accommodation and food service activities',
    'J Information and communication', 'K Financial and insurance',
    'L Real estate activities', 'M Knowledge-based services',
    'N Travel agent, cleaning, and other operationel services',
    'O Public administration, defence and compulsory social security',
    'P Education', 'Q Human health and social work',
    'R Arts, entertainment and recreation activities',
    'S Other service activities etc.'
])

subs = pd.Index([
    'A Agriculture, forestry and fishing', 'B Mining and quarrying',
    'CA Manufacture of food products, beverages and tobacco',
    'CB Textiles and leather products',
    'CC Wood and paper products and printing', 'CD Oil refinery etc.',
    'CE Manufacture of chemicals', 'CF Pharmaceuticals',
    'CG Manufacture of plastic, glass and concrete',
    'CH Basic metals and fabricated metal products',
    'CI Manufacture of electronic components',
    'CJ Electrical equipment', 'CK Manufacture of machinery',
    'CL Transport equipment',
    'CM Manufacture of furniture and other manufacturing etc.',
    'D Electricity, gas, steam and air conditioning supply',
    'E Water supply, sewerage and waste management', 'F Construction',
    'G Wholesale and retail trade', 'H Transportation',
    'I Accommodation and food service activities',
    'JA Publishing, television and radio broadcasting',
    'JB Telecommunications',
    'JC IT and information service activities',
    'K Financial and insurance',
    'LA Real estate activities and renting of non-residential buildings',
    'LB Dwellings', 'MA Consultancy etc.',
    'MB Scientific research and development',
    'MC Advertising and other business services',
    'N Travel agents, cleaning, and other operationel services',
    'O Public administration, defence and compulsory social security',
    # 'Of which: General government', 
    'P Education',
    'QA Human health activities', 'QB Residential care',
    'R Arts, entertainment and recreation activities',
    'SA Other service activities',
    'SB Activities of households as employers of domestic personnel'
])

# LIKE GGB
# Excluded industries
excluded_industries = {
    'Of': 'General government'  # Aggregate containing O, P, QA, QB - avoid double-counting
}

# Investment sector (capital-producing industries)
investment_industries = {
    'CC': 'Wood and paper products and printing',  
    'CG': 'Manufacture of plastic, glass and concrete',
    'CF': 'Pharmaceuticals',
    'CH': 'Basic metals and fabricated metal products',
    'CI': 'Manufacture of electronic components',
    'CJ': 'Electrical equipment',
    'CK': 'Manufacture of machinery',
    'CM': 'Manufacture of furniture and other manufacturing',
    'F': 'Construction',
    'MA': 'Consultancy etc.',
    'MB': 'Scientific research and development',
    'JC': 'IT and information service activities',
}

# Consumption sector (consumption-producing industries)
consumption_industries = {
    'A': 'Agriculture, forestry and fishing',
    'B': 'Mining and quarrying',
    'CA': 'Manufacture of food products, beverages and tobacco',
    'CB': 'Textiles and leather products',
    'CD': 'Oil refinery etc.', 
    'CE': 'Manufacture of chemicals',
    'CL': 'Transport equipment',
    'D': 'Electricity, gas, steam and air conditioning supply',
    'E': 'Water supply, sewerage and waste management',
    'G': 'Wholesale and retail trade',
    'H': 'Transportation',
    'I': 'Accommodation and food service activities',
    'JA': 'Publishing, television and radio broadcasting', 
    'JB': 'Telecommunications',
    'K': 'Financial and insurance',
    'LA': 'Real estate activities and renting of non-residential',
    'LB': 'Dwellings',
    'MC': 'Advertising and other business services',
    'N': 'Travel agents, cleaning, and other operational support services',
    'O': 'Public administration, defence and compulsory social security',
    'P': 'Education',
    'QA': 'Human health activities',
    'QB': 'Residential care',
    'R': 'Arts, entertainment and recreation activities',
    'SA': 'Other service activities',
    'SB': 'Activities of households as employers of domestic personnel'
}

# AGG ABOVE for direct
investment_industries2 = investment_industries.copy()
del investment_industries2['CC']
investment_industries2['JC'] = 'IT and information service activities'
investment_industries2['MA'] = 'Consultancy etc.'

consumption_industries2 = consumption_industries.copy()
consumption_industries2['CE'] = 'Manufacture of chemicals'
consumption_industries2['JC'] = 'IT and information service activities'
consumption_industries2['MA'] = 'Consultancy etc.'

# TOP 10
investment_industries3 = investment_industries2.copy()
del investment_industries2['CF'] 

consumption_industries3 = consumption_industries.copy()
del consumption_industries3['E']
del consumption_industries3['R']
del consumption_industries3['P']
del consumption_industries3['D']
del consumption_industries3['JB']
del consumption_industries3['CD']
del consumption_industries3['H']
del consumption_industries3['N']
del consumption_industries3['CB']
del consumption_industries3['MC']
del consumption_industries3['CL']
del consumption_industries3['LA']
del consumption_industries3['G']
del consumption_industries3['JA']

# NO PHARMA
investment_industries4 = investment_industries.copy()
del investment_industries4['CF']