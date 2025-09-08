###### Disease Dictionary

## "delete all "." as there are no "." in the database"
DISEASE_DEFS = {
    "sepsis": {
        "codes": ['99591', '99592', 
                '67024', 'A419', 'R6520', 
                'A4189', 'R6521', 'O0387', 
                'T8144XA', 'O0337', 'O0487', 
                'A4150', 'A410', 'A411', 'A412',
                    'A403', 'A414', 'A4151', 'A4152'],
        "keywords": ['sepsis']
    },
    "copd_exac": {
        "codes": ["J441", "J440", "49121", "49122"],
        "keywords": []
    },
    "acs_mi": {
        "codes": ["I200", "I210", "I211", "I212", "I213", "I214", "I219", "I220", "I221", "I222", "I228", "I229", "I230", "I231", "I232", "I233", "I234", "I235", "I236", "I237", "I240", "I241", "I248", "I249", "410", "4100", "4101", "4102", "4103", "4104", "4105", "4106", "4107", "4108", "4109", "4110", "4111", "41181", "41189"],
        "keywords": []
    },
    "stroke": {
        "codes": ["I60", "I61", "I62", "I63", "I64", "I65", "I66", "G45", "Z8673", "430", "431", "432", "43301", "43401", "43411", "436", "435", "V1254"],
        "keywords": ['stroke']
    },
    "ards": {
        "codes": ["J80", "J800", "51882"],
        "keywords": ['acute respiratory distress syndrome']
    },
    "aki": {
        "codes": ["N179", "N17","5849"],
        "keywords": ['acute kidney']
    },
    "pe": {
        "codes": ["I26", "I260", "I269", "41511", "41519"],
        "keywords": ['pulmonary embolism']
    },
    "pneumonia_bacterial": {
        "codes": ['J13', 'J14', 'J15', 'J150', 'J151', 'J152', 'J153', 'J154', 'J155', 'J156', 'J157', 'J158', 'J159'],
        "keywords": ['bacterial pneumonia']
    },
    "pneumonia_viral": {
        "codes": ['J09', 'J10', 'J11', 'J12', 'J120', 'J121', 'J122', 'J123', 'J128', 'J129'],
        "keywords": ['viral pneumonia']
    },
    "pneumonia_all": {
        "codes": ['J13', 'J14', 'J15', 'J150', 'J151', 'J152', 'J153', 'J154', 'J155', 'J156', 'J157', 'J158', 'J159','J09', 'J10', 'J11', 'J12', 'J120', 'J121', 'J122', 'J123', 'J128', 'J129'],
        "keywords": ['viral pneumonia', 'bacterial pneumonia']
    },
    "asthma_acute_exac": {
        "codes": ['J45901', 'J4521', 'J4531', 'J4541', 'J4551', '49392', '49301', '49302', '49311', '49312'],
        "keywords": []
    },
    "ahf": {
        "codes": ['I5021', 'I5023', 'I5031', 'I5033', 'I5041', 'I5043', 'I50811', 'I509', '42821', '42823', '42831', '42833', '42841', '42843', '4280'],
        "keywords": ['acute.*heart.*failure']
    }
}