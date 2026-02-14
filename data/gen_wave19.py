#!/usr/bin/env python3
"""Generate wave19_chemistry.json - exhaustive chemistry knowledge base."""
import json

concepts = []
relations = []

def C(label, definition, typ="FACT", trust=0.97):
    concepts.append({"label": label, "definition": definition, "type": typ, "trust": trust})

def R(source, target, typ="RELATES_TO", weight=0.9):
    relations.append({"source": source, "target": target, "type": typ, "weight": weight})

# ============================================================
# 1. ALL 118 ELEMENTS
# ============================================================
elements = [
    (1,"Hydrogen","H","Nonmetal","1.008"),
    (2,"Helium","He","Noble gas","4.003"),
    (3,"Lithium","Li","Alkali metal","6.941"),
    (4,"Beryllium","Be","Alkaline earth metal","9.012"),
    (5,"Boron","B","Metalloid","10.81"),
    (6,"Carbon","C","Nonmetal","12.011"),
    (7,"Nitrogen","N","Nonmetal","14.007"),
    (8,"Oxygen","O","Nonmetal","15.999"),
    (9,"Fluorine","F","Halogen","18.998"),
    (10,"Neon","Ne","Noble gas","20.180"),
    (11,"Sodium","Na","Alkali metal","22.990"),
    (12,"Magnesium","Mg","Alkaline earth metal","24.305"),
    (13,"Aluminium","Al","Post-transition metal","26.982"),
    (14,"Silicon","Si","Metalloid","28.086"),
    (15,"Phosphorus","P","Nonmetal","30.974"),
    (16,"Sulfur","S","Nonmetal","32.065"),
    (17,"Chlorine","Cl","Halogen","35.453"),
    (18,"Argon","Ar","Noble gas","39.948"),
    (19,"Potassium","K","Alkali metal","39.098"),
    (20,"Calcium","Ca","Alkaline earth metal","40.078"),
    (21,"Scandium","Sc","Transition metal","44.956"),
    (22,"Titanium","Ti","Transition metal","47.867"),
    (23,"Vanadium","V","Transition metal","50.942"),
    (24,"Chromium","Cr","Transition metal","51.996"),
    (25,"Manganese","Mn","Transition metal","54.938"),
    (26,"Iron","Fe","Transition metal","55.845"),
    (27,"Cobalt","Co","Transition metal","58.933"),
    (28,"Nickel","Ni","Transition metal","58.693"),
    (29,"Copper","Cu","Transition metal","63.546"),
    (30,"Zinc","Zn","Transition metal","65.38"),
    (31,"Gallium","Ga","Post-transition metal","69.723"),
    (32,"Germanium","Ge","Metalloid","72.630"),
    (33,"Arsenic","As","Metalloid","74.922"),
    (34,"Selenium","Se","Nonmetal","78.971"),
    (35,"Bromine","Br","Halogen","79.904"),
    (36,"Krypton","Kr","Noble gas","83.798"),
    (37,"Rubidium","Rb","Alkali metal","85.468"),
    (38,"Strontium","Sr","Alkaline earth metal","87.62"),
    (39,"Yttrium","Y","Transition metal","88.906"),
    (40,"Zirconium","Zr","Transition metal","91.224"),
    (41,"Niobium","Nb","Transition metal","92.906"),
    (42,"Molybdenum","Mo","Transition metal","95.95"),
    (43,"Technetium","Tc","Transition metal","98"),
    (44,"Ruthenium","Ru","Transition metal","101.07"),
    (45,"Rhodium","Rh","Transition metal","102.91"),
    (46,"Palladium","Pd","Transition metal","106.42"),
    (47,"Silver","Ag","Transition metal","107.87"),
    (48,"Cadmium","Cd","Transition metal","112.41"),
    (49,"Indium","In","Post-transition metal","114.82"),
    (50,"Tin","Sn","Post-transition metal","118.71"),
    (51,"Antimony","Sb","Metalloid","121.76"),
    (52,"Tellurium","Te","Metalloid","127.60"),
    (53,"Iodine","I","Halogen","126.90"),
    (54,"Xenon","Xe","Noble gas","131.29"),
    (55,"Caesium","Cs","Alkali metal","132.91"),
    (56,"Barium","Ba","Alkaline earth metal","137.33"),
    (57,"Lanthanum","La","Lanthanide","138.91"),
    (58,"Cerium","Ce","Lanthanide","140.12"),
    (59,"Praseodymium","Pr","Lanthanide","140.91"),
    (60,"Neodymium","Nd","Lanthanide","144.24"),
    (61,"Promethium","Pm","Lanthanide","145"),
    (62,"Samarium","Sm","Lanthanide","150.36"),
    (63,"Europium","Eu","Lanthanide","151.96"),
    (64,"Gadolinium","Gd","Lanthanide","157.25"),
    (65,"Terbium","Tb","Lanthanide","158.93"),
    (66,"Dysprosium","Dy","Lanthanide","162.50"),
    (67,"Holmium","Ho","Lanthanide","164.93"),
    (68,"Erbium","Er","Lanthanide","167.26"),
    (69,"Thulium","Tm","Lanthanide","168.93"),
    (70,"Ytterbium","Yb","Lanthanide","173.05"),
    (71,"Lutetium","Lu","Lanthanide","174.97"),
    (72,"Hafnium","Hf","Transition metal","178.49"),
    (73,"Tantalum","Ta","Transition metal","180.95"),
    (74,"Tungsten","W","Transition metal","183.84"),
    (75,"Rhenium","Re","Transition metal","186.21"),
    (76,"Osmium","Os","Transition metal","190.23"),
    (77,"Iridium","Ir","Transition metal","192.22"),
    (78,"Platinum","Pt","Transition metal","195.08"),
    (79,"Gold","Au","Transition metal","196.97"),
    (80,"Mercury","Hg","Transition metal","200.59"),
    (81,"Thallium","Tl","Post-transition metal","204.38"),
    (82,"Lead","Pb","Post-transition metal","207.2"),
    (83,"Bismuth","Bi","Post-transition metal","208.98"),
    (84,"Polonium","Po","Post-transition metal","209"),
    (85,"Astatine","At","Halogen","210"),
    (86,"Radon","Rn","Noble gas","222"),
    (87,"Francium","Fr","Alkali metal","223"),
    (88,"Radium","Ra","Alkaline earth metal","226"),
    (89,"Actinium","Ac","Actinide","227"),
    (90,"Thorium","Th","Actinide","232.04"),
    (91,"Protactinium","Pa","Actinide","231.04"),
    (92,"Uranium","U","Actinide","238.03"),
    (93,"Neptunium","Np","Actinide","237"),
    (94,"Plutonium","Pu","Actinide","244"),
    (95,"Americium","Am","Actinide","243"),
    (96,"Curium","Cm","Actinide","247"),
    (97,"Berkelium","Bk","Actinide","247"),
    (98,"Californium","Cf","Actinide","251"),
    (99,"Einsteinium","Es","Actinide","252"),
    (100,"Fermium","Fm","Actinide","257"),
    (101,"Mendelevium","Md","Actinide","258"),
    (102,"Nobelium","No","Actinide","259"),
    (103,"Lawrencium","Lr","Actinide","266"),
    (104,"Rutherfordium","Rf","Transition metal","267"),
    (105,"Dubnium","Db","Transition metal","268"),
    (106,"Seaborgium","Sg","Transition metal","269"),
    (107,"Bohrium","Bh","Transition metal","270"),
    (108,"Hassium","Hs","Transition metal","277"),
    (109,"Meitnerium","Mt","Unknown","278"),
    (110,"Darmstadtium","Ds","Unknown","281"),
    (111,"Roentgenium","Rg","Unknown","282"),
    (112,"Copernicium","Cn","Unknown","285"),
    (113,"Nihonium","Nh","Unknown","286"),
    (114,"Flerovium","Fl","Unknown","289"),
    (115,"Moscovium","Mc","Unknown","290"),
    (116,"Livermorium","Lv","Unknown","293"),
    (117,"Tennessine","Ts","Unknown","294"),
    (118,"Oganesson","Og","Unknown","294"),
]

C("Periodic Table", "Tabular arrangement of chemical elements ordered by atomic number", "DEFINITION", 0.99)
C("Chemical Element", "Pure substance consisting of one type of atom distinguished by atomic number", "DEFINITION", 0.99)

for z, name, sym, cat, mass in elements:
    C(name, f"Chemical element with symbol {sym}, atomic number {z}, atomic mass {mass} u, classified as {cat}", "FACT", 0.99)
    R(name, "Chemical Element", "IS_A", 0.99)
    R(name, "Periodic Table", "PART_OF", 0.95)

# Element categories
categories = ["Alkali metal","Alkaline earth metal","Transition metal","Post-transition metal",
              "Metalloid","Nonmetal","Halogen","Noble gas","Lanthanide","Actinide"]
for cat in categories:
    C(cat, f"{cat} - a category of chemical elements in the periodic table", "DEFINITION", 0.97)
    R(cat, "Periodic Table", "PART_OF", 0.9)

for z, name, sym, cat, mass in elements:
    if cat in categories:
        R(name, cat, "IS_A", 0.95)

# ============================================================
# 2. COMPOUNDS (500+)
# ============================================================
compounds = [
    ("Water","H2O","Dihydrogen monoxide, essential solvent for life, bp 100°C"),
    ("Carbon Dioxide","CO2","Colorless gas produced by combustion and respiration"),
    ("Sodium Chloride","NaCl","Common table salt, ionic compound"),
    ("Hydrochloric Acid","HCl","Strong mineral acid, highly corrosive"),
    ("Sulfuric Acid","H2SO4","Strong diprotic acid, widely used industrial chemical"),
    ("Nitric Acid","HNO3","Strong oxidizing acid used in fertilizers and explosives"),
    ("Phosphoric Acid","H3PO4","Triprotic acid used in food and fertilizers"),
    ("Acetic Acid","CH3COOH","Weak organic acid, main component of vinegar"),
    ("Sodium Hydroxide","NaOH","Caustic soda, strong base used in industry"),
    ("Potassium Hydroxide","KOH","Caustic potash, strong base"),
    ("Calcium Hydroxide","Ca(OH)2","Slaked lime, used in construction"),
    ("Ammonia","NH3","Pungent gas, key industrial chemical for fertilizers"),
    ("Methane","CH4","Simplest hydrocarbon, main component of natural gas"),
    ("Ethane","C2H6","Two-carbon alkane, component of natural gas"),
    ("Propane","C3H8","Three-carbon alkane, used as fuel"),
    ("Butane","C4H10","Four-carbon alkane, used in lighters"),
    ("Pentane","C5H12","Five-carbon alkane, volatile liquid"),
    ("Hexane","C6H14","Six-carbon alkane, common lab solvent"),
    ("Heptane","C7H16","Seven-carbon alkane, reference fuel"),
    ("Octane","C8H18","Eight-carbon alkane, gasoline component"),
    ("Ethylene","C2H4","Simplest alkene, major industrial chemical for polyethylene"),
    ("Propylene","C3H6","Three-carbon alkene, used in polypropylene production"),
    ("Acetylene","C2H2","Simplest alkyne, used in welding"),
    ("Benzene","C6H6","Simplest aromatic hydrocarbon, carcinogenic solvent"),
    ("Toluene","C7H8","Methylbenzene, common industrial solvent"),
    ("Xylene","C8H10","Dimethylbenzene, solvent mixture"),
    ("Phenol","C6H5OH","Hydroxybenzene, used in plastics production"),
    ("Aniline","C6H5NH2","Aminobenzene, precursor to dyes"),
    ("Methanol","CH3OH","Simplest alcohol, toxic, used as solvent and fuel"),
    ("Ethanol","C2H5OH","Drinking alcohol, biofuel, solvent"),
    ("Propanol","C3H7OH","Three-carbon alcohol"),
    ("Isopropanol","(CH3)2CHOH","Rubbing alcohol, common disinfectant"),
    ("Butanol","C4H9OH","Four-carbon alcohol, potential biofuel"),
    ("Glycerol","C3H8O3","Triol used in pharmaceuticals and food"),
    ("Ethylene Glycol","C2H6O2","Diol used as antifreeze"),
    ("Formaldehyde","HCHO","Simplest aldehyde, used in resins and preservation"),
    ("Acetaldehyde","CH3CHO","Two-carbon aldehyde, metabolite of ethanol"),
    ("Acetone","CH3COCH3","Simplest ketone, common solvent"),
    ("Formic Acid","HCOOH","Simplest carboxylic acid, found in ant venom"),
    ("Oxalic Acid","H2C2O4","Dicarboxylic acid found in many plants"),
    ("Citric Acid","C6H8O7","Tricarboxylic acid found in citrus fruits"),
    ("Lactic Acid","C3H6O3","Alpha-hydroxy acid produced during anaerobic metabolism"),
    ("Tartaric Acid","C4H6O6","Diprotic acid found in grapes"),
    ("Benzoic Acid","C6H5COOH","Simplest aromatic carboxylic acid, food preservative"),
    ("Salicylic Acid","C7H6O3","Beta-hydroxy acid, precursor to aspirin"),
    ("Aspirin","C9H8O4","Acetylsalicylic acid, common analgesic"),
    ("Urea","(NH2)2CO","Organic compound, main nitrogen-containing waste in mammals"),
    ("Glucose","C6H12O6","Simple sugar, primary energy source for cells"),
    ("Fructose","C6H12O6","Fruit sugar, ketohexose isomer of glucose"),
    ("Sucrose","C12H22O11","Table sugar, disaccharide of glucose and fructose"),
    ("Lactose","C12H22O11","Milk sugar, disaccharide of glucose and galactose"),
    ("Maltose","C12H22O11","Malt sugar, disaccharide of two glucose units"),
    ("Starch","(C6H10O5)n","Polysaccharide energy storage in plants"),
    ("Cellulose","(C6H10O5)n","Polysaccharide structural component of plant cell walls"),
    ("Calcium Carbonate","CaCO3","Limestone, chalk, marble, shells"),
    ("Sodium Bicarbonate","NaHCO3","Baking soda"),
    ("Potassium Permanganate","KMnO4","Strong oxidizing agent, purple color"),
    ("Hydrogen Peroxide","H2O2","Oxidizer and bleaching agent"),
    ("Sodium Hypochlorite","NaClO","Active ingredient in bleach"),
    ("Calcium Oxide","CaO","Quicklime, produced by calcination of limestone"),
    ("Silicon Dioxide","SiO2","Silica, main component of sand and quartz"),
    ("Aluminium Oxide","Al2O3","Alumina, corundum, very hard ceramic"),
    ("Iron(III) Oxide","Fe2O3","Rust, hematite, red pigment"),
    ("Iron(II,III) Oxide","Fe3O4","Magnetite, naturally magnetic iron oxide"),
    ("Titanium Dioxide","TiO2","White pigment in paints and sunscreen"),
    ("Zinc Oxide","ZnO","Used in sunscreen and rubber manufacturing"),
    ("Copper(II) Sulfate","CuSO4","Blue vitriol, used as fungicide"),
    ("Silver Nitrate","AgNO3","Used in photography and as antiseptic"),
    ("Lead(II) Acetate","Pb(C2H3O2)2","Sugar of lead, historically toxic sweetener"),
    ("Mercury(II) Chloride","HgCl2","Corrosive sublimate, highly toxic"),
    ("Potassium Dichromate","K2Cr2O7","Strong oxidizing agent, orange crystals"),
    ("Sodium Sulfate","Na2SO4","Glauber's salt, used in detergents"),
    ("Magnesium Sulfate","MgSO4","Epsom salt, used in bath salts and medicine"),
    ("Calcium Sulfate","CaSO4","Gypsum, used in plaster and drywall"),
    ("Barium Sulfate","BaSO4","Used as radiocontrast agent in medical imaging"),
    ("Sodium Carbonate","Na2CO3","Washing soda, used in glass making"),
    ("Potassium Carbonate","K2CO3","Potash, used in soap making"),
    ("Magnesium Carbonate","MgCO3","Magnesite, used in fireproofing"),
    ("Lithium Carbonate","Li2CO3","Used to treat bipolar disorder"),
    ("Sodium Nitrate","NaNO3","Chile saltpeter, used in fertilizers"),
    ("Potassium Nitrate","KNO3","Saltpeter, used in gunpowder and fertilizers"),
    ("Ammonium Nitrate","NH4NO3","Fertilizer and explosive component"),
    ("Sodium Phosphate","Na3PO4","Used in cleaning agents and food"),
    ("Calcium Phosphate","Ca3(PO4)2","Main component of bones and teeth"),
    ("Ammonium Sulfate","(NH4)2SO4","Common nitrogen fertilizer"),
    ("Potassium Chloride","KCl","Sylvite, used as salt substitute and fertilizer"),
    ("Calcium Chloride","CaCl2","Desiccant and de-icing agent"),
    ("Magnesium Chloride","MgCl2","Found in seawater, used in tofu making"),
    ("Zinc Chloride","ZnCl2","Flux for soldering, used in batteries"),
    ("Iron(III) Chloride","FeCl3","Used in PCB etching and water treatment"),
    ("Aluminium Chloride","AlCl3","Friedel-Crafts catalyst"),
    ("Tin(II) Chloride","SnCl2","Reducing agent, used in tin plating"),
    ("Sodium Fluoride","NaF","Used in toothpaste and water fluoridation"),
    ("Calcium Fluoride","CaF2","Fluorite mineral, optical material"),
    ("Hydrogen Fluoride","HF","Highly corrosive acid, used in glass etching"),
    ("Hydrogen Sulfide","H2S","Toxic gas with rotten egg smell"),
    ("Sulfur Dioxide","SO2","Volcanic gas, used as preservative (E220)"),
    ("Sulfur Trioxide","SO3","Precursor to sulfuric acid"),
    ("Nitrogen Dioxide","NO2","Brown toxic gas, air pollutant"),
    ("Nitric Oxide","NO","Signaling molecule in biology, air pollutant"),
    ("Nitrous Oxide","N2O","Laughing gas, anesthetic and greenhouse gas"),
    ("Carbon Monoxide","CO","Toxic colorless gas from incomplete combustion"),
    ("Ozone","O3","Triatomic oxygen, UV shield in stratosphere"),
    ("Chloroform","CHCl3","Trichloromethane, early anesthetic, solvent"),
    ("Carbon Tetrachloride","CCl4","Solvent, ozone-depleting substance"),
    ("Dichloromethane","CH2Cl2","Methylene chloride, common lab solvent"),
    ("Diethyl Ether","(C2H5)2O","Classic anesthetic, highly flammable solvent"),
    ("Tetrahydrofuran","C4H8O","THF, cyclic ether solvent"),
    ("Dimethyl Sulfoxide","(CH3)2SO","DMSO, polar aprotic solvent"),
    ("Dimethylformamide","(CH3)2NCHO","DMF, polar aprotic solvent"),
    ("Acetonitrile","CH3CN","Polar aprotic solvent used in HPLC"),
    ("Pyridine","C5H5N","Heterocyclic aromatic amine, solvent and reagent"),
    ("Naphthalene","C10H8","Bicyclic aromatic hydrocarbon, mothball ingredient"),
    ("Anthracene","C14H10","Tricyclic aromatic hydrocarbon"),
    ("Styrene","C8H8","Vinyl benzene, monomer for polystyrene"),
    ("Vinyl Chloride","C2H3Cl","Monomer for PVC, carcinogenic gas"),
    ("Acrylonitrile","C3H3N","Monomer for acrylic fibers"),
    ("Ethylene Oxide","C2H4O","Sterilant and precursor for ethylene glycol"),
    ("Propylene Oxide","C3H6O","Used in polyurethane production"),
    ("Acrylic Acid","C3H4O2","Monomer for polyacrylates"),
    ("Methyl Methacrylate","C5H8O2","Monomer for PMMA (Plexiglas)"),
    ("Caprolactam","C6H11NO","Monomer for Nylon-6"),
    ("Adipic Acid","C6H10O4","Monomer for Nylon-6,6"),
    ("Hexamethylenediamine","C6H16N2","Monomer for Nylon-6,6"),
    ("Terephthalic Acid","C8H6O4","Monomer for PET polyester"),
    ("Bisphenol A","C15H16O2","Monomer for polycarbonate and epoxy resins"),
    ("Phosgene","COCl2","Toxic gas used in polycarbonate synthesis"),
    ("Isocyanate","R-NCO","Functional group class used in polyurethanes"),
    ("Glycine","C2H5NO2","Simplest amino acid"),
    ("Alanine","C3H7NO2","Non-polar amino acid"),
    ("Valine","C5H11NO2","Essential branched-chain amino acid"),
    ("Leucine","C6H13NO2","Essential branched-chain amino acid"),
    ("Isoleucine","C6H13NO2","Essential branched-chain amino acid"),
    ("Proline","C5H9NO2","Cyclic amino acid"),
    ("Phenylalanine","C9H11NO2","Essential aromatic amino acid"),
    ("Tryptophan","C11H12N2O2","Essential amino acid, serotonin precursor"),
    ("Methionine","C5H11NO2S","Essential sulfur-containing amino acid"),
    ("Serine","C3H7NO3","Hydroxyl-containing amino acid"),
    ("Threonine","C4H9NO3","Essential hydroxyl amino acid"),
    ("Cysteine","C3H7NO2S","Thiol-containing amino acid, forms disulfide bonds"),
    ("Tyrosine","C9H11NO3","Aromatic amino acid, precursor to dopamine"),
    ("Asparagine","C4H8N2O3","Amide of aspartic acid"),
    ("Glutamine","C5H10N2O3","Most abundant amino acid in blood"),
    ("Aspartic Acid","C4H7NO4","Acidic amino acid"),
    ("Glutamic Acid","C5H9NO4","Acidic amino acid, umami flavor (MSG)"),
    ("Lysine","C6H14N2O2","Essential basic amino acid"),
    ("Arginine","C6H14N4O2","Basic amino acid, NO precursor"),
    ("Histidine","C6H9N3O2","Essential amino acid with imidazole group"),
    ("Adenine","C5H5N5","Purine nucleobase in DNA and RNA"),
    ("Guanine","C5H5N5O","Purine nucleobase in DNA and RNA"),
    ("Cytosine","C4H5N3O","Pyrimidine nucleobase in DNA and RNA"),
    ("Thymine","C5H6N2O2","Pyrimidine nucleobase in DNA only"),
    ("Uracil","C4H4N2O2","Pyrimidine nucleobase in RNA only"),
    ("ATP","C10H16N5O13P3","Adenosine triphosphate, cellular energy currency"),
    ("NAD+","C21H27N7O14P2","Nicotinamide adenine dinucleotide, electron carrier"),
    ("Cholesterol","C27H46O","Sterol lipid essential for cell membranes"),
    ("Caffeine","C8H10N4O2","Stimulant alkaloid found in coffee and tea"),
    ("Nicotine","C10H14N2","Addictive alkaloid in tobacco"),
    ("Morphine","C17H19NO3","Opioid alkaloid analgesic from opium poppy"),
    ("Quinine","C20H24N2O2","Antimalarial alkaloid from cinchona bark"),
    ("Capsaicin","C18H27NO3","Compound responsible for chili pepper heat"),
    ("Vanillin","C8H8O3","Primary flavor compound in vanilla"),
    ("Limonene","C10H16","Terpene giving citrus its scent"),
    ("Menthol","C10H20O","Cooling compound from mint"),
    ("Camphor","C10H16O","Terpenoid with strong aroma, used medicinally"),
    ("Retinol","C20H30O","Vitamin A, essential for vision"),
    ("Ascorbic Acid","C6H8O6","Vitamin C, antioxidant, prevents scurvy"),
    ("Thiamine","C12H17N4OS","Vitamin B1, essential coenzyme"),
    ("Riboflavin","C17H20N4O6","Vitamin B2, yellow pigment"),
    ("Niacin","C6H5NO2","Vitamin B3, precursor to NAD"),
    ("Penicillin","C16H18N2O4S","First widely used antibiotic, beta-lactam"),
    ("Ibuprofen","C13H18O2","NSAID anti-inflammatory drug"),
    ("Paracetamol","C8H9NO2","Acetaminophen, common analgesic and antipyretic"),
    ("DDT","C14H9Cl5","Organochlorine insecticide, now largely banned"),
    ("Glyphosate","C3H8NO5P","Broad-spectrum herbicide"),
    ("Chlorophyll","C55H72MgN4O5","Green pigment in plants enabling photosynthesis"),
    ("Hemoglobin","complex","Iron-containing oxygen-transport metalloprotein in red blood cells"),
    ("Insulin","protein","Peptide hormone regulating blood glucose"),
    ("DNA","polymer","Deoxyribonucleic acid, carrier of genetic information"),
    ("RNA","polymer","Ribonucleic acid, involved in protein synthesis"),
    ("Sodium Thiosulfate","Na2S2O3","Used to neutralize chlorine and in photography"),
    ("Potassium Iodide","KI","Used as iodine supplement and radiation protectant"),
    ("Boric Acid","H3BO3","Weak acid used as antiseptic and insecticide"),
    ("Chromic Acid","H2CrO4","Strong oxidizing acid used in chrome plating"),
    ("Perchloric Acid","HClO4","Strongest common mineral acid"),
    ("Hydrobromic Acid","HBr","Strong acid used in organic synthesis"),
    ("Hydroiodic Acid","HI","Strong acid, strongest of hydrohalic acids"),
    ("Hypochlorous Acid","HClO","Weak acid, disinfectant in swimming pools"),
    ("Chlorous Acid","HClO2","Weak acid used as bleaching agent"),
    ("Chloric Acid","HClO3","Strong acid, oxidizing agent"),
    ("Carbonic Acid","H2CO3","Weak diprotic acid formed when CO2 dissolves in water"),
    ("Silicic Acid","H4SiO4","Weak acid from silicate dissolution"),
    ("Phosphorous Acid","H3PO3","Diprotic reducing acid"),
    ("Arsenic Acid","H3AsO4","Triprotic acid, toxic"),
    ("Selenic Acid","H2SeO4","Strong acid analogous to sulfuric acid"),
    ("Telluric Acid","Te(OH)6","Hexaprotic weak acid"),
    ("Chromium(III) Oxide","Cr2O3","Green pigment, abrasive"),
    ("Manganese Dioxide","MnO2","Used in batteries and as oxidizing agent"),
    ("Vanadium Pentoxide","V2O5","Catalyst in contact process"),
    ("Cobalt(II) Chloride","CoCl2","Humidity indicator, turns blue when dry"),
    ("Nickel(II) Sulfate","NiSO4","Used in nickel plating"),
    ("Copper(I) Oxide","Cu2O","Red copper oxide, semiconductor"),
    ("Gold(III) Chloride","AuCl3","Used as catalyst in organic chemistry"),
    ("Platinum(IV) Chloride","PtCl4","Catalyst precursor"),
    ("Palladium(II) Chloride","PdCl2","Wacker process catalyst"),
    ("Ruthenium Tetroxide","RuO4","Strong oxidizing agent"),
    ("Osmium Tetroxide","OsO4","Dihydroxylation reagent, highly toxic"),
    ("Tungsten Carbide","WC","Extremely hard material for cutting tools"),
    ("Boron Nitride","BN","Ceramic with diamond-like hardness (cubic form)"),
    ("Silicon Carbide","SiC","Carborundum, abrasive and semiconductor"),
    ("Gallium Arsenide","GaAs","III-V semiconductor for electronics"),
    ("Indium Tin Oxide","In2O3·SnO2","Transparent conductor for touchscreens"),
    ("Lithium Cobalt Oxide","LiCoO2","Cathode material in lithium-ion batteries"),
    ("Lithium Iron Phosphate","LiFePO4","Cathode material for LFP batteries"),
    ("Sodium Azide","NaN3","Airbag propellant, highly toxic"),
    ("Potassium Cyanide","KCN","Extremely toxic, used in gold extraction"),
    ("Sodium Cyanide","NaCN","Used in gold mining, highly toxic"),
    ("Calcium Carbide","CaC2","Produces acetylene when reacted with water"),
    ("Magnesium Hydroxide","Mg(OH)2","Milk of magnesia, antacid"),
    ("Aluminium Hydroxide","Al(OH)3","Amphoteric hydroxide, antacid"),
    ("Barium Hydroxide","Ba(OH)2","Strong base used in titrations"),
    ("Strontium Hydroxide","Sr(OH)2","Strong base"),
    ("Lithium Hydroxide","LiOH","Used in CO2 scrubbers in spacecraft"),
    ("Caesium Hydroxide","CsOH","Strongest alkali metal hydroxide"),
    ("Tetramethylsilane","Si(CH3)4","TMS, NMR reference standard"),
    ("Dimethyl Ether","CH3OCH3","Simplest ether, potential diesel substitute"),
    ("Ethyl Acetate","C4H8O2","Common solvent in nail polish remover"),
    ("Methyl Salicylate","C8H8O3","Oil of wintergreen"),
    ("Nitroglycerine","C3H5N3O9","Explosive, also used as vasodilator"),
    ("TNT","C7H5N3O6","Trinitrotoluene, common military explosive"),
    ("RDX","C3H6N6O6","Cyclotrimethylenetrinitramine, powerful explosive"),
    ("Sodium Lauryl Sulfate","C12H25NaO4S","Common surfactant in soaps"),
    ("EDTA","C10H16N2O8","Chelating agent used in analytical chemistry"),
    ("Crown Ether 18-Crown-6","C12H24O6","Macrocyclic polyether that binds potassium ions"),
    ("Ferrocene","Fe(C5H5)2","Organometallic sandwich compound"),
    ("Cisplatin","Pt(NH3)2Cl2","Platinum-based anticancer drug"),
    ("Grignard Reagent","RMgX","Organomagnesium halide used in C-C bond formation"),
    ("Lithium Aluminium Hydride","LiAlH4","Powerful reducing agent in organic chemistry"),
    ("Sodium Borohydride","NaBH4","Mild reducing agent for carbonyl compounds"),
    ("Potassium tert-Butoxide","(CH3)3COK","Strong non-nucleophilic base"),
    ("n-Butyllithium","C4H9Li","Strong base and nucleophile in organic synthesis"),
    ("Diazomethane","CH2N2","Methylating agent, explosive gas"),
    ("Thionyl Chloride","SOCl2","Reagent for converting carboxylic acids to acyl chlorides"),
    ("Phosphorus Pentachloride","PCl5","Chlorinating agent"),
    ("Phosphorus Trichloride","PCl3","Used in organophosphorus chemistry"),
    ("Sulfuryl Chloride","SO2Cl2","Chlorosulfonation reagent"),
    ("Borane","BH3","Lewis acid, hydroboration reagent"),
    ("Diborane","B2H6","Boron hydride used in organic synthesis"),
    ("Silane","SiH4","Silicon hydride used in semiconductor manufacturing"),
    ("Germane","GeH4","Germanium hydride for semiconductor deposition"),
    ("Phosphine","PH3","Toxic phosphorus hydride gas"),
    ("Arsine","AsH3","Extremely toxic arsenic hydride"),
    ("Stibine","SbH3","Antimony hydride, very toxic"),
    ("Trimethylamine","(CH3)3N","Fishy-smelling tertiary amine"),
    ("Triethylamine","(C2H5)3N","Common organic base"),
    ("Pyrrole","C4H5N","Five-membered aromatic heterocycle with nitrogen"),
    ("Furan","C4H4O","Five-membered aromatic heterocycle with oxygen"),
    ("Thiophene","C4H4S","Five-membered aromatic heterocycle with sulfur"),
    ("Imidazole","C3H4N2","Five-membered ring with two nitrogens"),
    ("Indole","C8H7N","Bicyclic aromatic heterocycle found in tryptophan"),
    ("Purine","C5H4N4","Bicyclic aromatic heterocycle, base for adenine/guanine"),
    ("Pyrimidine","C4H4N2","Six-membered ring with two nitrogens"),
    ("Piperidine","C5H11N","Saturated six-membered nitrogen heterocycle"),
    ("Morpholine","C4H9NO","Six-membered ring with O and N"),
    ("Biphenyl","C12H10","Two connected phenyl rings"),
    ("Fluorescein","C20H12O5","Fluorescent dye used as tracer"),
    ("Rhodamine B","C28H31ClN2O3","Red fluorescent dye"),
    ("Methylene Blue","C16H18ClN3S","Thiazine dye used as biological stain"),
    ("Phenolphthalein","C20H14O4","pH indicator, pink in base"),
    ("Litmus","complex","pH indicator from lichens, red/blue"),
    ("Bromothymol Blue","C27H28Br2O5S","pH indicator for near-neutral solutions"),
    ("Ninhydrin","C9H6O4","Reagent for detecting amino acids"),
    ("Benedict's Reagent","Cu2+ complex","Test for reducing sugars"),
    ("Fehling's Solution","Cu2+ tartrate","Test for aldehydes and reducing sugars"),
    ("Tollens' Reagent","Ag(NH3)2+","Silver mirror test for aldehydes"),
    ("Karl Fischer Reagent","I2/SO2/pyridine","Reagent for water content determination"),
    ("Nessler's Reagent","K2HgI4","Test for ammonia"),
    ("Sodium Acetate","CH3COONa","Used in heating pads and as buffer"),
    ("Ammonium Chloride","NH4Cl","Used in soldering flux and cough medicine"),
    ("Potassium Bromide","KBr","Used as anticonvulsant and in photography"),
    ("Sodium Iodide","NaI","Scintillation detector material"),
    ("Calcium Hypochlorite","Ca(ClO)2","Swimming pool chlorinator"),
    ("Aluminium Sulfate","Al2(SO4)3","Used in water purification"),
    ("Ferrous Sulfate","FeSO4","Iron supplement, mordant"),
    ("Copper(II) Nitrate","Cu(NO3)2","Blue crystalline solid, used in patinas"),
    ("Zinc Sulfate","ZnSO4","Used in dietary supplements"),
    ("Lead(II) Nitrate","Pb(NO3)2","Oxidizer, used in gold cyanidation"),
    ("Mercury(I) Chloride","Hg2Cl2","Calomel, historical medicine"),
    ("Tin(IV) Oxide","SnO2","Cassiterite, tin ore"),
    ("Antimony Trioxide","Sb2O3","Flame retardant"),
    ("Bismuth Subsalicylate","C7H5BiO4","Active ingredient in Pepto-Bismol"),
    ("Cerium(IV) Oxide","CeO2","Used in catalytic converters and polishing"),
    ("Neodymium Oxide","Nd2O3","Used in coloring glass purple"),
    ("Lanthanum Oxide","La2O3","Used in optical glass"),
    ("Yttrium Oxide","Y2O3","Used in YAG lasers"),
    ("Zirconium Dioxide","ZrO2","Zirconia, used in dental crowns"),
    ("Hafnium Dioxide","HfO2","High-k dielectric in semiconductors"),
    ("Niobium Pentoxide","Nb2O5","Used in optical glass"),
    ("Tantalum Pentoxide","Ta2O5","Used in capacitors"),
    ("Molybdenum Disulfide","MoS2","Solid lubricant"),
    ("Tungsten Disulfide","WS2","Solid lubricant, 2D material"),
    ("Rhenium Diboride","ReB2","Ultra-hard compound"),
    ("Uranium Hexafluoride","UF6","Used in uranium enrichment"),
    ("Plutonium Dioxide","PuO2","Nuclear fuel form"),
    ("Thorium Dioxide","ThO2","Highest melting point of any oxide"),
    ("Beryllium Oxide","BeO","Excellent thermal conductor ceramic"),
    ("Magnesium Oxide","MgO","Magnesia, refractory material"),
    ("Strontium Titanate","SrTiO3","Perovskite with high dielectric constant"),
    ("Barium Titanate","BaTiO3","Piezoelectric ceramic"),
    ("Lead Zirconate Titanate","Pb(Zr,Ti)O3","PZT, widely used piezoelectric"),
    ("Lithium Niobate","LiNbO3","Nonlinear optical material"),
    ("Cadmium Sulfide","CdS","Yellow pigment, semiconductor"),
    ("Cadmium Selenide","CdSe","Quantum dot material"),
    ("Zinc Selenide","ZnSe","Infrared optical material"),
    ("Gallium Nitride","GaN","Wide-bandgap semiconductor for LEDs"),
    ("Aluminium Nitride","AlN","Thermally conductive ceramic substrate"),
    ("Titanium Nitride","TiN","Gold-colored hard coating"),
    ("Chromium Nitride","CrN","Hard coating for cutting tools"),
    ("Iron Pyrite","FeS2","Fool's gold"),
    ("Copper(I) Sulfide","Cu2S","Chalcocite, copper ore"),
    ("Lead(II) Sulfide","PbS","Galena, lead ore"),
    ("Zinc Sulfide","ZnS","Sphalerite/wurtzite, phosphor material"),
    ("Molybdenite","MoS2","Primary molybdenum ore"),
    ("Cinnabar","HgS","Mercury ore, red pigment"),
    ("Stibnite","Sb2S3","Antimony ore"),
    ("Realgar","As4S4","Arsenic sulfide mineral"),
    ("Orpiment","As2S3","Yellow arsenic sulfide mineral"),
    ("Sodium Peroxide","Na2O2","Strong oxidizer, used in O2 generators"),
    ("Potassium Superoxide","KO2","Used in rebreathers to generate oxygen"),
    ("Barium Peroxide","BaO2","Used in pyrotechnics"),
    ("Ammonium Perchlorate","NH4ClO4","Solid rocket propellant oxidizer"),
    ("Potassium Chlorate","KClO3","Oxidizer in matches and pyrotechnics"),
    ("Sodium Chlorate","NaClO3","Herbicide and pulp bleaching agent"),
    ("Calcium Cyanamide","CaCN2","Nitrogen fertilizer, defoliant"),
    ("Thiourea","(NH2)2CS","Sulfur analog of urea, photographic fixing"),
    ("Dimethyl Carbonate","C3H6O3","Green solvent, methylating agent"),
    ("Propylene Glycol","C3H8O2","Antifreeze, food additive (E1520)"),
    ("Polyethylene Glycol","H(OCH2CH2)nOH","PEG, used in laxatives and drug delivery"),
    ("Dimethylacetamide","C4H9NO","DMAc, polar aprotic solvent"),
    ("N-Methyl-2-pyrrolidone","C5H9NO","NMP, high-boiling solvent"),
    ("Hexamethylphosphoramide","(Me2N)3PO","HMPA, powerful polar aprotic solvent (carcinogenic)"),
    ("1,4-Dioxane","C4H8O2","Cyclic ether solvent"),
    ("Petrol Ether","mixture","Low-boiling hydrocarbon mixture used as solvent"),
    ("Tert-Butyl Methyl Ether","C5H12O","MTBE, fuel oxygenate"),
    ("Ethyl Formate","C3H6O2","Ester with rum-like aroma"),
    ("Butyl Acetate","C6H12O2","Solvent with fruity odor"),
    ("Methyl Benzoate","C8H8O2","Ester with wintergreen-like odor"),
    ("Acetic Anhydride","(CH3CO)2O","Acetylating agent"),
    ("Maleic Anhydride","C4H2O3","Dienophile in Diels-Alder reactions"),
    ("Phthalic Anhydride","C8H4O3","Precursor for phthalate plasticizers"),
    ("Succinic Anhydride","C4H4O3","Used in organic synthesis"),
    ("Trifluoroacetic Acid","CF3COOH","TFA, strong organic acid, HPLC solvent"),
    ("Methanesulfonic Acid","CH3SO3H","Strong organic acid catalyst"),
    ("p-Toluenesulfonic Acid","CH3C6H4SO3H","TsOH, common acid catalyst"),
    ("Triflic Acid","CF3SO3H","One of strongest known acids"),
    ("Fluoroantimonic Acid","HSbF6","Strongest known superacid"),
    ("Magic Acid","FSO3H·SbF5","Superacid dissolving hydrocarbons"),
    ("1-Bromobutane","C4H9Br","Primary alkyl halide, SN2 substrate"),
    ("2-Bromobutane","C4H9Br","Secondary alkyl halide, SN1/SN2"),
    ("tert-Butyl Chloride","(CH3)3CCl","Tertiary alkyl halide, SN1 substrate"),
    ("Benzyl Chloride","C6H5CH2Cl","Benzylic halide"),
    ("Allyl Bromide","C3H5Br","Allylic halide"),
    ("Epichlorohydrin","C3H5ClO","Epoxy resin precursor"),
    ("Glycidol","C3H6O2","Epoxide alcohol, polymer precursor"),
    ("Catechol","C6H6O2","1,2-Benzenediol"),
    ("Resorcinol","C6H6O2","1,3-Benzenediol"),
    ("Hydroquinone","C6H6O2","1,4-Benzenediol, photographic developer"),
    ("p-Benzoquinone","C6H4O2","Oxidized form of hydroquinone"),
    ("Cyclohexane","C6H12","Cycloalkane, non-polar solvent"),
    ("Cyclohexanone","C6H10O","Precursor to caprolactam/nylon"),
    ("Cyclopentadiene","C5H6","Diene for Diels-Alder, precursor to ferrocene"),
    ("1,3-Butadiene","C4H6","Conjugated diene, synthetic rubber monomer"),
    ("Isoprene","C5H8","Monomer of natural rubber"),
    ("Chloroprene","C4H5Cl","Monomer of neoprene rubber"),
    ("Tetrafluoroethylene","C2F4","Monomer of Teflon (PTFE)"),
    ("Vinylidene Fluoride","C2H2F2","Monomer of PVDF"),
    ("Hexafluoropropylene","C3F6","Comonomer for fluoropolymers"),
    ("Ethylene Diamine","C2H8N2","Chelating ligand precursor"),
    ("Diethylenetriamine","C4H13N3","DETA, epoxy curing agent"),
    ("Melamine","C3H6N6","Triazine used in resins and laminates"),
    ("Uric Acid","C5H4N4O3","Purine degradation product, causes gout"),
    ("Creatinine","C4H7N3O","Waste product of muscle metabolism"),
    ("Dopamine","C8H11NO2","Neurotransmitter involved in reward"),
    ("Serotonin","C10H12N2O","Neurotransmitter regulating mood"),
    ("Adrenaline","C9H13NO3","Epinephrine, fight-or-flight hormone"),
    ("Noradrenaline","C8H11NO3","Norepinephrine, stress hormone and neurotransmitter"),
    ("Acetylcholine","C7H16NO2","Neurotransmitter at neuromuscular junctions"),
    ("GABA","C4H9NO2","Gamma-aminobutyric acid, inhibitory neurotransmitter"),
    ("Glutathione","C10H17N3O6S","Tripeptide antioxidant"),
    ("Coenzyme A","C21H36N7O16P3S","CoA, central to metabolism"),
    ("FAD","C27H33N9O15P2","Flavin adenine dinucleotide, electron carrier"),
    ("Heme","C34H32FeN4O4","Iron porphyrin complex in hemoglobin"),
    ("Chlorophyll a","C55H72MgN4O5","Primary photosynthetic pigment"),
    ("Beta-Carotene","C40H56","Orange pigment, provitamin A"),
    ("Lycopene","C40H56","Red pigment in tomatoes"),
    ("Anthocyanin","C15H11O+","Class of plant pigments (red/blue/purple)"),
    ("Taxol","C47H51NO14","Paclitaxel, anticancer drug from yew tree"),
    ("Artemisinin","C15H22O5","Antimalarial from sweet wormwood"),
    ("Resveratrol","C14H12O3","Polyphenol antioxidant in red wine"),
    ("Curcumin","C21H20O6","Yellow compound from turmeric"),
    ("Theobromine","C7H8N4O2","Alkaloid in chocolate"),
    ("Theophylline","C7H8N4O2","Bronchodilator found in tea"),
]

for label, formula, defn in compounds:
    C(label, f"{defn} (formula: {formula})", "FACT", 0.96)
    R(label, "Chemical Compound", "IS_A", 0.9)

C("Chemical Compound", "Substance composed of two or more elements chemically bonded in fixed proportions", "DEFINITION", 0.99)

# Element-compound relations (selected key ones)
element_compound_links = [
    ("Water","Hydrogen"),("Water","Oxygen"),
    ("Carbon Dioxide","Carbon"),("Carbon Dioxide","Oxygen"),
    ("Sodium Chloride","Sodium"),("Sodium Chloride","Chlorine"),
    ("Ammonia","Nitrogen"),("Ammonia","Hydrogen"),
    ("Methane","Carbon"),("Methane","Hydrogen"),
    ("Sulfuric Acid","Sulfur"),("Sulfuric Acid","Oxygen"),("Sulfuric Acid","Hydrogen"),
    ("Hydrochloric Acid","Hydrogen"),("Hydrochloric Acid","Chlorine"),
    ("Ethanol","Carbon"),("Ethanol","Hydrogen"),("Ethanol","Oxygen"),
    ("Glucose","Carbon"),("Glucose","Hydrogen"),("Glucose","Oxygen"),
    ("Calcium Carbonate","Calcium"),("Calcium Carbonate","Carbon"),("Calcium Carbonate","Oxygen"),
    ("Iron(III) Oxide","Iron"),("Iron(III) Oxide","Oxygen"),
    ("Silicon Dioxide","Silicon"),("Silicon Dioxide","Oxygen"),
    ("Titanium Dioxide","Titanium"),("Titanium Dioxide","Oxygen"),
    ("Sodium Hydroxide","Sodium"),("Sodium Hydroxide","Oxygen"),("Sodium Hydroxide","Hydrogen"),
]
for comp, elem in element_compound_links:
    R(elem, comp, "PART_OF", 0.95)

# ============================================================
# 3. FUNCTIONAL GROUPS (35+)
# ============================================================
functional_groups = [
    ("Hydroxyl Group","-OH","Characteristic of alcohols and phenols"),
    ("Carboxyl Group","-COOH","Characteristic of carboxylic acids"),
    ("Amino Group","-NH2","Characteristic of amines and amino acids"),
    ("Carbonyl Group","C=O","Found in aldehydes, ketones, and other compounds"),
    ("Aldehyde Group","-CHO","Terminal carbonyl group"),
    ("Ketone Group","R-CO-R'","Internal carbonyl group"),
    ("Ester Group","-COOR","Product of acid-alcohol condensation"),
    ("Ether Group","R-O-R'","Oxygen between two carbon groups"),
    ("Amide Group","-CONH2","Peptide bond functional group"),
    ("Thiol Group","-SH","Sulfhydryl, characteristic of thiols"),
    ("Sulfide Group","R-S-R'","Thioether linkage"),
    ("Disulfide Group","R-S-S-R'","Important in protein structure"),
    ("Nitro Group","-NO2","Found in explosives and dyes"),
    ("Nitrile Group","-CN","Cyano group"),
    ("Isocyanate Group","-NCO","Used in polyurethane chemistry"),
    ("Azide Group","-N3","Energetic functional group"),
    ("Azo Group","-N=N-","Characteristic of azo dyes"),
    ("Phosphate Group","-OPO3²⁻","Found in DNA, ATP, phospholipids"),
    ("Sulfonate Group","-SO3⁻","Strong acid group, surfactants"),
    ("Sulfonyl Group","-SO2-","Found in sulfonamide drugs"),
    ("Acyl Chloride Group","-COCl","Reactive acid derivative"),
    ("Anhydride Group","-(CO)O(CO)-","Reactive acid derivative"),
    ("Epoxide Group","3-membered C-O-C ring","Strained cyclic ether"),
    ("Lactone Group","cyclic ester","Intramolecular ester"),
    ("Lactam Group","cyclic amide","Found in penicillin (beta-lactam)"),
    ("Imine Group","C=N","Schiff base linkage"),
    ("Oxime Group","C=NOH","Formed from carbonyl + hydroxylamine"),
    ("Hydrazone Group","C=NNH2","Formed from carbonyl + hydrazine"),
    ("Enol Group","C=C-OH","Tautomer of carbonyl"),
    ("Acetal Group","R-CH(OR')2","Protected aldehyde form"),
    ("Hemiacetal Group","R-CH(OH)(OR')","Intermediate in acetal formation"),
    ("Vinyl Group","CH2=CH-","Terminal alkene"),
    ("Allyl Group","CH2=CH-CH2-","Allylic system"),
    ("Phenyl Group","C6H5-","Aromatic ring substituent"),
    ("Benzyl Group","C6H5CH2-","Methylene attached to phenyl"),
    ("Methyl Group","-CH3","Simplest alkyl group"),
    ("Ethyl Group","-C2H5","Two-carbon alkyl group"),
    ("Isopropyl Group","(CH3)2CH-","Branched three-carbon group"),
    ("tert-Butyl Group","(CH3)3C-","Bulky four-carbon group"),
]

C("Functional Group", "Specific group of atoms within molecules responsible for characteristic chemical reactions", "DEFINITION", 0.99)
for name, structure, defn in functional_groups:
    C(name, f"{defn} (structure: {structure})", "DEFINITION", 0.96)
    R(name, "Functional Group", "IS_A", 0.95)
    R(name, "Organic Chemistry", "PART_OF", 0.85)

# ============================================================
# 4. REACTION TYPES
# ============================================================
C("Organic Chemistry", "Branch of chemistry studying carbon-containing compounds", "DEFINITION", 0.99)
C("Inorganic Chemistry", "Branch of chemistry studying non-carbon compounds and metals", "DEFINITION", 0.99)
C("Physical Chemistry", "Branch applying physics to chemical systems", "DEFINITION", 0.99)
C("Analytical Chemistry", "Branch focused on determining composition of substances", "DEFINITION", 0.99)
C("Biochemistry", "Chemistry of biological systems and processes", "DEFINITION", 0.99)

reactions = [
    ("Combustion","Reaction with oxygen producing heat, light, CO2, and H2O"),
    ("Acid-Base Neutralization","Reaction between acid and base producing salt and water"),
    ("Oxidation-Reduction (Redox)","Reaction involving transfer of electrons between species"),
    ("Single Displacement","One element replaces another in a compound"),
    ("Double Displacement","Exchange of ions between two compounds"),
    ("Decomposition Reaction","Breakdown of a compound into simpler substances"),
    ("Synthesis Reaction","Combination of two or more substances to form a product"),
    ("Precipitation Reaction","Formation of insoluble solid from mixing solutions"),
    ("Hydrolysis","Cleavage of chemical bonds by addition of water"),
    ("Condensation Reaction","Two molecules combine with loss of small molecule (usually water)"),
    ("Addition Reaction","Atoms add across a multiple bond"),
    ("Elimination Reaction","Removal of atoms to form a multiple bond"),
    ("Substitution Reaction","One atom or group replaced by another"),
    ("SN1 Reaction","Unimolecular nucleophilic substitution via carbocation intermediate"),
    ("SN2 Reaction","Bimolecular nucleophilic substitution, backside attack, inversion"),
    ("E1 Elimination","Unimolecular elimination via carbocation"),
    ("E2 Elimination","Bimolecular elimination, concerted, anti-periplanar"),
    ("Electrophilic Addition","Addition of electrophile to double bond"),
    ("Electrophilic Aromatic Substitution","Substitution on aromatic ring by electrophile"),
    ("Nucleophilic Aromatic Substitution","Substitution on aromatic ring by nucleophile"),
    ("Radical Substitution","Free radical chain reaction replacing atom"),
    ("Radical Addition","Free radical adding across multiple bond"),
    ("Diels-Alder Reaction","[4+2] cycloaddition of diene and dienophile"),
    ("Claisen Rearrangement","[3,3]-sigmatropic rearrangement of allyl vinyl ethers"),
    ("Cope Rearrangement","[3,3]-sigmatropic rearrangement of 1,5-dienes"),
    ("Wittig Reaction","Formation of alkene from aldehyde/ketone and phosphonium ylide"),
    ("Grignard Reaction","Organomagnesium addition to carbonyl"),
    ("Aldol Reaction","Condensation of two carbonyls forming β-hydroxy carbonyl"),
    ("Claisen Condensation","Condensation of two esters forming β-keto ester"),
    ("Michael Addition","Conjugate addition of nucleophile to α,β-unsaturated carbonyl"),
    ("Friedel-Crafts Alkylation","Alkyl group added to aromatic ring with Lewis acid catalyst"),
    ("Friedel-Crafts Acylation","Acyl group added to aromatic ring with Lewis acid catalyst"),
    ("Fischer Esterification","Acid-catalyzed formation of ester from acid and alcohol"),
    ("Williamson Ether Synthesis","SN2 formation of ether from alkoxide and alkyl halide"),
    ("Suzuki Coupling","Pd-catalyzed cross-coupling of boronic acid with aryl halide"),
    ("Heck Reaction","Pd-catalyzed coupling of aryl halide with alkene"),
    ("Sonogashira Coupling","Pd/Cu-catalyzed coupling of aryl halide with terminal alkyne"),
    ("Stille Coupling","Pd-catalyzed coupling with organotin compounds"),
    ("Negishi Coupling","Pd-catalyzed coupling with organozinc compounds"),
    ("Buchwald-Hartwig Amination","Pd-catalyzed C-N bond formation"),
    ("Olefin Metathesis","Metal-catalyzed redistribution of alkene fragments"),
    ("Hydrogenation","Addition of H2 across unsaturated bonds"),
    ("Dehydrogenation","Removal of H2 to form unsaturation"),
    ("Halogenation","Introduction of halogen atoms"),
    ("Hydrohalogenation","Addition of HX across multiple bond"),
    ("Dehydrohalogenation","Elimination of HX to form alkene"),
    ("Hydration","Addition of water across multiple bond"),
    ("Dehydration","Removal of water, e.g. alcohol to alkene"),
    ("Epoxidation","Formation of epoxide from alkene"),
    ("Dihydroxylation","Addition of two OH groups to alkene"),
    ("Ozonolysis","Cleavage of alkene by ozone"),
    ("Beckmann Rearrangement","Oxime to amide rearrangement"),
    ("Hofmann Rearrangement","Amide to amine with loss of one carbon"),
    ("Curtius Rearrangement","Acyl azide to isocyanate"),
    ("Pinacol Rearrangement","1,2-diol to ketone under acid conditions"),
    ("Wagner-Meerwein Rearrangement","1,2-shift of alkyl or aryl group to adjacent carbocation"),
    ("Baeyer-Villiger Oxidation","Ketone to ester/lactone using peracid"),
    ("Swern Oxidation","Alcohol to aldehyde/ketone using DMSO/oxalyl chloride"),
    ("Jones Oxidation","Alcohol to carboxylic acid/ketone with chromic acid"),
    ("Birch Reduction","Partial reduction of aromatic ring with Na/NH3"),
    ("Wolff-Kishner Reduction","Carbonyl to methylene using hydrazine/base"),
    ("Clemmensen Reduction","Carbonyl to methylene using Zn-Hg/HCl"),
    ("Cannizzaro Reaction","Disproportionation of aldehyde to acid + alcohol"),
    ("Tischenko Reaction","Aldehyde dimerization to ester"),
    ("Mannich Reaction","Three-component reaction: aldehyde + amine + enolizable carbonyl"),
    ("Strecker Synthesis","Amino acid synthesis from aldehyde + ammonia + cyanide"),
    ("Gabriel Synthesis","Primary amine synthesis using phthalimide"),
    ("Reimer-Tiemann Reaction","Formylation of phenol with CHCl3/NaOH"),
    ("Kolbe-Schmitt Reaction","Carboxylation of phenoxide with CO2"),
    ("Sandmeyer Reaction","Replacement of diazonium group with halide via Cu catalyst"),
    ("Balz-Schiemann Reaction","Fluorination via diazonium tetrafluoroborate"),
    ("Hell-Volhard-Zelinsky Reaction","Alpha-bromination of carboxylic acids"),
    ("Arndt-Eistert Synthesis","Homologation of carboxylic acid by one carbon"),
    ("Ring-Closing Metathesis","Intramolecular olefin metathesis forming cyclic alkene"),
    ("Click Chemistry (CuAAC)","Cu-catalyzed azide-alkyne cycloaddition"),
    ("Polymerization","Repeated addition/condensation forming macromolecules"),
    ("Transesterification","Exchange of ester alkoxy group"),
    ("Saponification","Base hydrolysis of ester to carboxylate salt"),
    ("Fermentation","Anaerobic conversion of sugars to ethanol and CO2"),
    ("Photosynthesis","Light-driven synthesis of glucose from CO2 and H2O"),
    ("Haber Process","Industrial synthesis of ammonia from N2 and H2"),
    ("Contact Process","Industrial synthesis of sulfuric acid"),
    ("Ostwald Process","Industrial oxidation of ammonia to nitric acid"),
    ("Solvay Process","Industrial production of sodium carbonate"),
    ("Fischer-Tropsch Process","Conversion of syngas to hydrocarbons"),
    ("Wacker Process","Pd-catalyzed oxidation of ethylene to acetaldehyde"),
    ("Ziegler-Natta Polymerization","Stereospecific polymerization of alkenes"),
    ("ATRP","Atom Transfer Radical Polymerization for controlled polymers"),
    ("RAFT Polymerization","Reversible Addition-Fragmentation chain Transfer polymerization"),
    ("Electroplating","Electrodeposition of metal coating"),
    ("Electrolysis","Driving non-spontaneous reactions with electric current"),
    ("Galvanic Cell Reaction","Spontaneous redox reaction generating electricity"),
    ("Corrosion","Electrochemical degradation of metals"),
    ("Calcination","Thermal decomposition of carbonates/hydrates"),
    ("Roasting","Heating ore in air to convert sulfides to oxides"),
    ("Smelting","Extracting metal from ore using heat and reducing agent"),
]

C("Chemical Reaction", "Process of transforming reactants into products", "DEFINITION", 0.99)
for name, defn in reactions:
    C(name, defn, "DEFINITION", 0.97)
    R(name, "Chemical Reaction", "IS_A", 0.9)

# Link some reactions to organic/inorganic
organic_rxns = ["SN1 Reaction","SN2 Reaction","E1 Elimination","E2 Elimination",
    "Diels-Alder Reaction","Grignard Reaction","Aldol Reaction","Wittig Reaction",
    "Suzuki Coupling","Heck Reaction","Friedel-Crafts Alkylation","Friedel-Crafts Acylation",
    "Fischer Esterification","Michael Addition","Electrophilic Aromatic Substitution"]
for r in organic_rxns:
    R(r, "Organic Chemistry", "PART_OF", 0.85)

industrial_rxns = ["Haber Process","Contact Process","Ostwald Process","Solvay Process",
    "Fischer-Tropsch Process","Wacker Process","Ziegler-Natta Polymerization"]
for r in industrial_rxns:
    R(r, "Industrial Chemistry", "PART_OF", 0.85)
C("Industrial Chemistry","Application of chemistry to large-scale manufacturing","DEFINITION",0.95)

# ============================================================
# 5. POLYMERS (55+)
# ============================================================
polymers = [
    ("Polyethylene","PE","Most produced plastic, from ethylene monomer"),
    ("High-Density Polyethylene","HDPE","Linear PE, rigid, used in bottles and pipes"),
    ("Low-Density Polyethylene","LDPE","Branched PE, flexible, used in plastic bags"),
    ("Linear Low-Density Polyethylene","LLDPE","Copolymer PE with short branches"),
    ("Ultra-High-Molecular-Weight Polyethylene","UHMWPE","Extremely tough PE for bearings and armor"),
    ("Polypropylene","PP","From propylene, used in packaging and textiles"),
    ("Polyvinyl Chloride","PVC","From vinyl chloride, used in pipes and flooring"),
    ("Polystyrene","PS","From styrene, used in packaging and insulation"),
    ("Expanded Polystyrene","EPS","Styrofoam, lightweight insulation material"),
    ("Acrylonitrile Butadiene Styrene","ABS","Tough terpolymer used in 3D printing and electronics"),
    ("Polyethylene Terephthalate","PET","Polyester for bottles and clothing fibers"),
    ("Polybutylene Terephthalate","PBT","Engineering thermoplastic"),
    ("Nylon-6","PA6","Polyamide from caprolactam"),
    ("Nylon-6,6","PA66","Polyamide from adipic acid and hexamethylenediamine"),
    ("Nylon-12","PA12","Polyamide used in 3D printing (SLS)"),
    ("Kevlar","aromatic polyamide","Para-aramid fiber with extreme tensile strength"),
    ("Nomex","aromatic polyamide","Meta-aramid fiber, heat and flame resistant"),
    ("Polytetrafluoroethylene","PTFE","Teflon, non-stick fluoropolymer"),
    ("Polyvinylidene Fluoride","PVDF","Fluoropolymer with piezoelectric properties"),
    ("Polyurethane","PU","Versatile polymer for foams, coatings, elastomers"),
    ("Polycarbonate","PC","Transparent tough plastic from bisphenol A"),
    ("Polymethyl Methacrylate","PMMA","Plexiglas/Lucite, transparent thermoplastic"),
    ("Polyoxymethylene","POM","Delrin/acetal, engineering thermoplastic"),
    ("Polyether Ether Ketone","PEEK","High-performance semi-crystalline thermoplastic"),
    ("Polyimide","PI","Kapton, heat-resistant polymer for electronics"),
    ("Polysulfone","PSU","High-temperature engineering plastic"),
    ("Polyether Sulfone","PES","Transparent high-temp polymer"),
    ("Polyphenylene Sulfide","PPS","Chemical-resistant engineering plastic"),
    ("Polyphenylene Oxide","PPO","Engineering thermoplastic, often blended"),
    ("Polyetherimide","PEI","Ultem, high-performance amorphous thermoplastic"),
    ("Polylactic Acid","PLA","Biodegradable polyester from lactic acid"),
    ("Polyhydroxybutyrate","PHB","Biodegradable polyester from bacteria"),
    ("Polyglycolic Acid","PGA","Biodegradable polyester for sutures"),
    ("Polyvinyl Acetate","PVAc","Used in wood glue and latex paints"),
    ("Polyvinyl Alcohol","PVA","Water-soluble synthetic polymer"),
    ("Polyacrylamide","PAM","Used in gel electrophoresis and water treatment"),
    ("Polyacrylonitrile","PAN","Precursor to carbon fiber"),
    ("Polyisoprene","natural rubber","cis-1,4-Polyisoprene, natural rubber"),
    ("Polybutadiene","BR","Synthetic rubber for tires"),
    ("Styrene-Butadiene Rubber","SBR","Most widely used synthetic rubber"),
    ("Nitrile Rubber","NBR","Oil-resistant rubber from acrylonitrile-butadiene"),
    ("Neoprene","CR","Polychloroprene, chemical-resistant rubber"),
    ("Silicone Rubber","PDMS","Polysiloxane elastomer, heat resistant"),
    ("Ethylene Propylene Diene Rubber","EPDM","Weather-resistant rubber for seals"),
    ("Butyl Rubber","IIR","Low gas permeability, used in inner tubes"),
    ("Epoxy Resin","thermoset","Cross-linked polymer for adhesives and composites"),
    ("Phenol-Formaldehyde Resin","Bakelite","First synthetic plastic (1907)"),
    ("Urea-Formaldehyde Resin","UF","Used in particle board adhesive"),
    ("Melamine-Formaldehyde Resin","MF","Used in laminates (Formica)"),
    ("Unsaturated Polyester Resin","UPR","Used in fiberglass composites"),
    ("Vinyl Ester Resin","VER","Improved corrosion-resistant composite resin"),
    ("Polyurethane Foam","PU foam","Flexible/rigid foam for insulation and cushions"),
    ("Cellulose Acetate","CA","Semi-synthetic polymer for film and fibers"),
    ("Cellophane","regenerated cellulose","Transparent film from cellulose"),
    ("Rayon","regenerated cellulose","Semi-synthetic fiber"),
    ("Vulcanized Rubber","cross-linked NR","Sulfur-crosslinked natural rubber (Goodyear 1839)"),
]

C("Polymer", "Large molecule composed of repeating structural units (monomers)", "DEFINITION", 0.99)
C("Thermoplastic", "Polymer that softens on heating and can be remolded", "DEFINITION", 0.97)
C("Thermoset", "Polymer that irreversibly hardens when cured", "DEFINITION", 0.97)
C("Elastomer", "Polymer with elastic properties (rubber-like)", "DEFINITION", 0.97)

for name, abbr, defn in polymers:
    C(name, f"{defn} (abbrev: {abbr})", "FACT", 0.96)
    R(name, "Polymer", "IS_A", 0.95)

# Classify polymers
thermoplastics = ["Polyethylene","Polypropylene","Polyvinyl Chloride","Polystyrene",
    "Polyethylene Terephthalate","Nylon-6","Nylon-6,6","Polycarbonate","PMMA","PEEK","PLA"]
for p in thermoplastics:
    nm = p if p in [x[0] for x in polymers] else None
    if nm: R(nm, "Thermoplastic", "IS_A", 0.9)

thermosets = ["Epoxy Resin","Phenol-Formaldehyde Resin","Urea-Formaldehyde Resin","Melamine-Formaldehyde Resin"]
for p in thermosets:
    R(p, "Thermoset", "IS_A", 0.9)

elastomers = ["Polyisoprene","Polybutadiene","Styrene-Butadiene Rubber","Nitrile Rubber",
    "Neoprene","Silicone Rubber","Butyl Rubber","Vulcanized Rubber"]
for p in elastomers:
    R(p, "Elastomer", "IS_A", 0.9)

# Polymethyl Methacrylate fix
R("Polymethyl Methacrylate", "Thermoplastic", "IS_A", 0.9)
R("Polyether Ether Ketone", "Thermoplastic", "IS_A", 0.9)
R("Polylactic Acid", "Thermoplastic", "IS_A", 0.9)

# ============================================================
# 6. CRYSTAL STRUCTURES
# ============================================================
crystals = [
    ("Face-Centered Cubic","FCC","Cubic close-packed, e.g., Cu, Al, Au, Ag"),
    ("Body-Centered Cubic","BCC","e.g., Fe (α), W, Cr, Na"),
    ("Hexagonal Close-Packed","HCP","e.g., Mg, Ti, Zn, Co"),
    ("Simple Cubic","SC","Rare, e.g., Polonium"),
    ("Diamond Cubic","DC","e.g., C (diamond), Si, Ge"),
    ("Rock Salt Structure","NaCl type","e.g., NaCl, KCl, MgO"),
    ("Cesium Chloride Structure","CsCl type","e.g., CsCl, CsBr"),
    ("Zinc Blende Structure","ZnS sphalerite","e.g., ZnS, GaAs, InP"),
    ("Wurtzite Structure","ZnS wurtzite","e.g., ZnO, GaN, AlN"),
    ("Fluorite Structure","CaF2 type","e.g., CaF2, UO2, ThO2"),
    ("Antifluorite Structure","inverse CaF2","e.g., Li2O, Na2O"),
    ("Rutile Structure","TiO2 type","e.g., TiO2, SnO2, MnO2"),
    ("Perovskite Structure","ABX3","e.g., BaTiO3, SrTiO3, CaTiO3"),
    ("Spinel Structure","AB2O4","e.g., MgAl2O4, Fe3O4"),
    ("Corundum Structure","Al2O3 type","e.g., Al2O3, Cr2O3, Fe2O3"),
    ("Garnet Structure","A3B2(SiO4)3","e.g., pyrope, almandine, grossular"),
    ("Olivine Structure","(Mg,Fe)2SiO4","Nesosilicate island structure"),
    ("Layered Structure","van der Waals layers","e.g., graphite, MoS2, BN"),
    ("Monoclinic Crystal System","one oblique axis","β ≠ 90°"),
    ("Triclinic Crystal System","all angles unequal","Lowest symmetry crystal system"),
    ("Orthorhombic Crystal System","three unequal axes at 90°","e.g., sulfur, aragonite"),
    ("Tetragonal Crystal System","a=b≠c, all 90°","e.g., rutile, zircon"),
    ("Hexagonal Crystal System","a=b≠c, γ=120°","e.g., quartz, beryl"),
    ("Cubic Crystal System","a=b=c, all 90°","Highest symmetry"),
    ("Trigonal Crystal System","rhombohedral","e.g., calcite, corundum"),
]

C("Crystal Structure","Regular arrangement of atoms in a crystalline solid","DEFINITION",0.99)
for name, abbr, defn in crystals:
    C(name, f"{defn} ({abbr})", "DEFINITION", 0.96)
    R(name, "Crystal Structure", "IS_A" if "System" not in name else "PART_OF", 0.9)

# Link elements to structures
R("Copper","Face-Centered Cubic","HAS_PROPERTY",0.95)
R("Gold","Face-Centered Cubic","HAS_PROPERTY",0.95)
R("Silver","Face-Centered Cubic","HAS_PROPERTY",0.95)
R("Aluminium","Face-Centered Cubic","HAS_PROPERTY",0.95)
R("Iron","Body-Centered Cubic","HAS_PROPERTY",0.95)
R("Tungsten","Body-Centered Cubic","HAS_PROPERTY",0.95)
R("Titanium","Hexagonal Close-Packed","HAS_PROPERTY",0.95)
R("Magnesium","Hexagonal Close-Packed","HAS_PROPERTY",0.95)
R("Carbon","Diamond Cubic","HAS_PROPERTY",0.9)
R("Silicon","Diamond Cubic","HAS_PROPERTY",0.95)

# ============================================================
# 7. ACIDS AND BASES (100+)
# ============================================================
C("Acid","Substance that donates protons (Brønsted) or accepts electron pairs (Lewis)","DEFINITION",0.99)
C("Base","Substance that accepts protons (Brønsted) or donates electron pairs (Lewis)","DEFINITION",0.99)
C("Strong Acid","Acid that completely dissociates in water","DEFINITION",0.98)
C("Weak Acid","Acid that partially dissociates in water","DEFINITION",0.98)
C("Strong Base","Base that completely dissociates in water","DEFINITION",0.98)
C("Weak Base","Base that partially dissociates in water","DEFINITION",0.98)
C("Superacid","Acid stronger than pure sulfuric acid","DEFINITION",0.97)
C("Lewis Acid","Electron pair acceptor","DEFINITION",0.98)
C("Lewis Base","Electron pair donor","DEFINITION",0.98)

# Already have many acids/bases as compounds, add classification relations
strong_acids = ["Hydrochloric Acid","Sulfuric Acid","Nitric Acid","Perchloric Acid",
    "Hydrobromic Acid","Hydroiodic Acid"]
for a in strong_acids:
    R(a, "Strong Acid", "IS_A", 0.95)
    R(a, "Acid", "IS_A", 0.95)

weak_acids = ["Acetic Acid","Formic Acid","Carbonic Acid","Phosphoric Acid","Boric Acid",
    "Oxalic Acid","Citric Acid","Lactic Acid","Tartaric Acid","Benzoic Acid","Salicylic Acid",
    "Hydrofluoric Acid","Silicic Acid","Phosphorous Acid","Ascorbic Acid","Uric Acid",
    "Hypochlorous Acid","Chlorous Acid","Acrylic Acid","Trifluoroacetic Acid"]
# HF is weak but add it
C("Hydrofluoric Acid","Weak acid HF, extremely corrosive, dissolves glass","FACT",0.97)
R("Hydrofluoric Acid","Weak Acid","IS_A",0.95)
for a in weak_acids:
    R(a, "Weak Acid", "IS_A", 0.95)
    R(a, "Acid", "IS_A", 0.95)

superacids = ["Fluoroantimonic Acid","Magic Acid","Triflic Acid"]
for a in superacids:
    R(a, "Superacid", "IS_A", 0.95)

strong_bases = ["Sodium Hydroxide","Potassium Hydroxide","Calcium Hydroxide",
    "Barium Hydroxide","Strontium Hydroxide","Lithium Hydroxide","Caesium Hydroxide"]
for b in strong_bases:
    R(b, "Strong Base", "IS_A", 0.95)
    R(b, "Base", "IS_A", 0.95)

weak_bases = ["Ammonia","Trimethylamine","Triethylamine","Pyridine","Aniline"]
for b in weak_bases:
    R(b, "Weak Base", "IS_A", 0.95)
    R(b, "Base", "IS_A", 0.95)

# Additional acids not yet covered
extra_acids = [
    ("Chromic Acid","H2CrO4","Strong oxidizing acid"),
    ("Permanganic Acid","HMnO4","Unstable strong acid"),
    ("Periodic Acid","HIO4","Oxidizing acid for diol cleavage"),
    ("Iodic Acid","HIO3","Moderately strong acid"),
    ("Bromic Acid","HBrO3","Strong acid"),
    ("Sulfurous Acid","H2SO3","Weak diprotic acid from SO2 in water"),
    ("Nitrous Acid","HNO2","Weak acid, source of nitrosonium ion"),
    ("Cyanhydric Acid","HCN","Extremely toxic weak acid, prussic acid"),
    ("Thiocyanic Acid","HSCN","Weak acid with pseudohalide anion"),
    ("Fulminic Acid","HCNO","Unstable isomer of cyanic acid"),
    ("Cyanic Acid","HOCN","Weak acid, tautomer of isocyanic acid"),
    ("Fluorosulfuric Acid","FSO3H","Very strong acid used in superacid mixtures"),
    ("Hexafluorophosphoric Acid","HPF6","Strong acid with non-coordinating anion"),
    ("Tetrafluoroboric Acid","HBF4","Strong acid with weakly coordinating anion"),
    ("Picric Acid","C6H3N3O7","Trinitrophenol, strong organic acid and explosive"),
    ("Dichloroacetic Acid","CHCl2COOH","Moderately strong chlorinated acetic acid"),
    ("Trichloroacetic Acid","CCl3COOH","Strong organic acid used in dermatology"),
]
for name, formula, defn in extra_acids:
    C(name, f"{defn} (formula: {formula})", "FACT", 0.95)
    R(name, "Acid", "IS_A", 0.95)

extra_bases = [
    ("Magnesium Hydroxide","Mg(OH)2","Weak base, antacid (milk of magnesia)"),
    ("Aluminium Hydroxide","Al(OH)3","Amphoteric hydroxide, used as antacid"),
    ("Sodium Carbonate","Na2CO3","Weak base in solution, washing soda"),
    ("Sodium Bicarbonate","NaHCO3","Mild base, baking soda"),
    ("Potassium Carbonate","K2CO3","Weak base, potash"),
    ("DBU","C9H16N2","1,8-Diazabicycloundec-7-ene, strong amidine base"),
    ("DMAP","C7H10N2","4-Dimethylaminopyridine, nucleophilic catalyst/base"),
    ("Sodium Hydride","NaH","Strong non-nucleophilic base"),
    ("Potassium tert-Butoxide","(CH3)3COK","Strong hindered base for E2"),
    ("LDA","LiN(iPr)2","Lithium diisopropylamide, strong non-nucleophilic base"),
    ("LHMDS","LiN(SiMe3)2","Lithium bis(trimethylsilyl)amide"),
    ("Guanidine","CH5N3","Strong organic base"),
    ("Piperidine","C5H11N","Secondary amine base"),
    ("Imidazole","C3H4N2","Heterocyclic base, histidine component"),
    ("Diethylamine","(C2H5)2NH","Secondary amine base"),
    ("Diisopropylamine","((CH3)2CH)2NH","Hindered secondary amine"),
    ("Morpholine","C4H9NO","Heterocyclic secondary amine base"),
    ("1,4-Diazabicyclo[2.2.2]octane","DABCO","Bicyclic tertiary amine base/catalyst"),
]
for name, formula, defn in extra_bases:
    if not any(c["label"]==name for c in concepts):
        C(name, f"{defn} (formula: {formula})", "FACT", 0.95)
    R(name, "Base", "IS_A", 0.95)

# ============================================================
# 8. SOLVENTS (55+)
# ============================================================
C("Solvent","Substance that dissolves a solute to form a solution","DEFINITION",0.99)
C("Polar Protic Solvent","Solvent with O-H or N-H bonds, can hydrogen bond","DEFINITION",0.97)
C("Polar Aprotic Solvent","Polar solvent without acidic hydrogens","DEFINITION",0.97)
C("Non-Polar Solvent","Solvent with low dielectric constant","DEFINITION",0.97)

solvents_list = [
    ("Water","Polar Protic Solvent"),
    ("Methanol","Polar Protic Solvent"),
    ("Ethanol","Polar Protic Solvent"),
    ("Isopropanol","Polar Protic Solvent"),
    ("Acetic Acid","Polar Protic Solvent"),
    ("Formic Acid","Polar Protic Solvent"),
    ("Ethylene Glycol","Polar Protic Solvent"),
    ("Glycerol","Polar Protic Solvent"),
    ("Acetone","Polar Aprotic Solvent"),
    ("Dimethyl Sulfoxide","Polar Aprotic Solvent"),
    ("Dimethylformamide","Polar Aprotic Solvent"),
    ("Acetonitrile","Polar Aprotic Solvent"),
    ("Tetrahydrofuran","Polar Aprotic Solvent"),
    ("Ethyl Acetate","Polar Aprotic Solvent"),
    ("Dichloromethane","Polar Aprotic Solvent"),
    ("Chloroform","Polar Aprotic Solvent"),
    ("Dimethylacetamide","Polar Aprotic Solvent"),
    ("N-Methyl-2-pyrrolidone","Polar Aprotic Solvent"),
    ("1,4-Dioxane","Polar Aprotic Solvent"),
    ("Hexane","Non-Polar Solvent"),
    ("Pentane","Non-Polar Solvent"),
    ("Cyclohexane","Non-Polar Solvent"),
    ("Benzene","Non-Polar Solvent"),
    ("Toluene","Non-Polar Solvent"),
    ("Xylene","Non-Polar Solvent"),
    ("Diethyl Ether","Non-Polar Solvent"),
    ("Carbon Tetrachloride","Non-Polar Solvent"),
    ("Heptane","Non-Polar Solvent"),
    ("Petrol Ether","Non-Polar Solvent"),
    ("Tert-Butyl Methyl Ether","Non-Polar Solvent"),
    ("Pyridine","Polar Aprotic Solvent"),
    ("Hexamethylphosphoramide","Polar Aprotic Solvent"),
    ("Dimethyl Carbonate","Polar Aprotic Solvent"),
]
for name, cat in solvents_list:
    R(name, "Solvent", "IS_A", 0.9)
    R(name, cat, "IS_A", 0.85)

# ============================================================
# 9. CATALYSTS
# ============================================================
C("Catalyst","Substance that increases reaction rate without being consumed","DEFINITION",0.99)
C("Homogeneous Catalyst","Catalyst in same phase as reactants","DEFINITION",0.97)
C("Heterogeneous Catalyst","Catalyst in different phase from reactants","DEFINITION",0.97)
C("Enzyme","Biological catalyst, usually protein","DEFINITION",0.99)
C("Enzyme Catalysis","Acceleration of reactions by enzymes with high specificity","DEFINITION",0.98)

catalysts_list = [
    ("Platinum Catalyst","Pt metal catalyst for hydrogenation, fuel cells, catalytic converters","Heterogeneous Catalyst"),
    ("Palladium Catalyst","Pd catalyst for cross-coupling, hydrogenation","Heterogeneous Catalyst"),
    ("Nickel Catalyst","Ni catalyst (Raney nickel) for hydrogenation","Heterogeneous Catalyst"),
    ("Iron Catalyst","Fe catalyst in Haber process for ammonia synthesis","Heterogeneous Catalyst"),
    ("Vanadium Pentoxide Catalyst","V2O5 in contact process for sulfuric acid","Heterogeneous Catalyst"),
    ("Rhodium Catalyst","Rh in catalytic converters and hydroformylation","Homogeneous Catalyst"),
    ("Ruthenium Catalyst","Ru in olefin metathesis (Grubbs)","Homogeneous Catalyst"),
    ("Iridium Catalyst","Ir in asymmetric hydrogenation","Homogeneous Catalyst"),
    ("Zeolite Catalyst","Microporous aluminosilicate for cracking","Heterogeneous Catalyst"),
    ("Aluminium Chloride Catalyst","AlCl3 Lewis acid for Friedel-Crafts","Homogeneous Catalyst"),
    ("Titanium Tetrachloride Catalyst","TiCl4 in Ziegler-Natta polymerization","Heterogeneous Catalyst"),
    ("Copper Catalyst","Cu in click chemistry and Ullmann coupling","Homogeneous Catalyst"),
    ("Manganese Dioxide Catalyst","MnO2 for H2O2 decomposition","Heterogeneous Catalyst"),
    ("Tin Catalyst","Sn compounds in polyurethane formation","Homogeneous Catalyst"),
    ("Acid Catalyst","Proton-donating catalyst (H+, H2SO4, TsOH)","Homogeneous Catalyst"),
    ("Base Catalyst","Hydroxide or amine catalyst","Homogeneous Catalyst"),
    ("Phase Transfer Catalyst","Facilitates reaction between immiscible phases","Homogeneous Catalyst"),
    ("Photocatalyst","Light-activated catalyst (TiO2, Ru(bpy)3)","Heterogeneous Catalyst"),
    ("Wilkinson's Catalyst","RhCl(PPh3)3, homogeneous hydrogenation","Homogeneous Catalyst"),
    ("Grubbs Catalyst","Ru carbene for olefin metathesis","Homogeneous Catalyst"),
    ("Crabtree's Catalyst","Ir complex for hindered alkene hydrogenation","Homogeneous Catalyst"),
    ("Lindlar's Catalyst","Pd/CaCO3/Pb, partial hydrogenation of alkynes to cis-alkenes","Heterogeneous Catalyst"),
    ("Adams' Catalyst","PtO2, reduces to Pt for hydrogenation","Heterogeneous Catalyst"),
    ("Sharpless Catalyst","Ti/tartrate for asymmetric epoxidation","Homogeneous Catalyst"),
    ("BINAP Catalyst","Chiral bisphosphine ligand for asymmetric synthesis","Homogeneous Catalyst"),
]

for name, defn, cat in catalysts_list:
    C(name, defn, "FACT", 0.96)
    R(name, "Catalyst", "IS_A", 0.95)
    R(name, cat, "IS_A", 0.9)

# Enzymes
enzymes = [
    ("Amylase","Enzyme that breaks down starch to sugars"),
    ("Lipase","Enzyme that hydrolyzes fats to fatty acids and glycerol"),
    ("Protease","Enzyme that cleaves peptide bonds in proteins"),
    ("DNA Polymerase","Enzyme that synthesizes DNA from nucleotides"),
    ("RNA Polymerase","Enzyme that synthesizes RNA from DNA template"),
    ("ATP Synthase","Enzyme that produces ATP from ADP and phosphate"),
    ("Catalase","Enzyme that decomposes H2O2 to water and oxygen"),
    ("Lactase","Enzyme that breaks down lactose to glucose and galactose"),
    ("Cellulase","Enzyme that breaks down cellulose"),
    ("Pepsin","Stomach protease active at low pH"),
    ("Trypsin","Pancreatic serine protease"),
    ("Lysozyme","Enzyme that breaks down bacterial cell walls"),
    ("Carbonic Anhydrase","Fastest known enzyme, interconverts CO2 and HCO3⁻"),
    ("Cytochrome P450","Family of oxidase enzymes for drug metabolism"),
    ("Reverse Transcriptase","Enzyme that transcribes RNA to DNA"),
    ("Restriction Enzyme","Enzyme that cuts DNA at specific sequences"),
    ("Ligase","Enzyme that joins DNA fragments"),
    ("Helicase","Enzyme that unwinds DNA double helix"),
    ("Kinase","Enzyme that transfers phosphate groups"),
    ("Phosphatase","Enzyme that removes phosphate groups"),
]
for name, defn in enzymes:
    C(name, defn, "FACT", 0.97)
    R(name, "Enzyme", "IS_A", 0.95)
    R(name, "Catalyst", "IS_A", 0.9)

# ============================================================
# 10. ALLOYS (55+)
# ============================================================
C("Alloy","Mixture of metals or metal with other elements","DEFINITION",0.99)

alloys = [
    ("Steel","Iron-carbon alloy (0.2-2.1% C), most important structural metal"),
    ("Stainless Steel","Steel with >10.5% Cr, corrosion resistant"),
    ("Carbon Steel","Steel with primarily carbon as alloying element"),
    ("Tool Steel","Hard steel alloy for cutting tools"),
    ("High-Speed Steel","Steel alloy (W, Mo, Cr, V) that retains hardness at high temperature"),
    ("Maraging Steel","Ultra-high strength steel with Ni, Co, Mo"),
    ("Damascus Steel","Historical steel with distinctive watered pattern"),
    ("Cast Iron","Iron with >2.1% carbon, brittle but castable"),
    ("Wrought Iron","Nearly pure iron, very low carbon, historically worked"),
    ("Bronze","Copper-tin alloy, historically first alloy"),
    ("Brass","Copper-zinc alloy, gold-colored"),
    ("Cupronickel","Copper-nickel alloy, used in coins"),
    ("Beryllium Copper","Cu-Be alloy, non-sparking tools"),
    ("Phosphor Bronze","Cu-Sn-P alloy, springs and bearings"),
    ("Aluminium Bronze","Cu-Al alloy, corrosion resistant"),
    ("Manganese Bronze","Cu-Zn-Mn alloy, marine hardware"),
    ("Pewter","Tin-based alloy (historically with lead)"),
    ("Solder","Tin-lead or tin-silver-copper alloy for joining"),
    ("Duralumin","Al-Cu alloy, aircraft construction"),
    ("Aluminium 6061","Al-Mg-Si alloy, general purpose"),
    ("Aluminium 7075","Al-Zn alloy, high strength aerospace"),
    ("Zamak","Zinc-aluminium alloy for die casting"),
    ("Nichrome","Ni-Cr alloy, heating elements"),
    ("Inconel","Ni-Cr superalloy, high temperature applications"),
    ("Monel","Ni-Cu alloy, corrosion resistant"),
    ("Hastelloy","Ni-Mo-Cr superalloy, chemical resistance"),
    ("Invar","Fe-Ni alloy with very low thermal expansion"),
    ("Kovar","Fe-Ni-Co alloy matching glass expansion"),
    ("Mu-Metal","Ni-Fe alloy with very high magnetic permeability"),
    ("Permalloy","Ni-Fe (80/20) alloy, high magnetic permeability"),
    ("Alnico","Al-Ni-Co alloy for permanent magnets"),
    ("Neodymium Magnet Alloy","Nd2Fe14B, strongest permanent magnets"),
    ("Samarium Cobalt Magnet","SmCo5, high-temperature permanent magnets"),
    ("Titanium 6Al-4V","Ti alloy, most common Ti alloy, aerospace/medical"),
    ("Nitinol","Ni-Ti shape memory alloy"),
    ("Stellite","Co-Cr-W alloy, extremely wear resistant"),
    ("Vitallium","Co-Cr-Mo alloy, medical implants"),
    ("Zircaloy","Zr alloy for nuclear fuel cladding"),
    ("Magnalium","Al-Mg alloy, lightweight and corrosion resistant"),
    ("Elektron","Mg-based alloy, lightest structural metal"),
    ("Babbitt Metal","Sn-Sb-Cu bearing alloy"),
    ("Type Metal","Pb-Sn-Sb alloy for printing"),
    ("Wood's Metal","Bi-Pb-Sn-Cd eutectic, mp 70°C"),
    ("Rose's Metal","Bi-Pb-Sn eutectic, mp 94°C"),
    ("Galinstan","Ga-In-Sn eutectic, liquid at room temperature"),
    ("Amalgam","Mercury alloy, historically used in dentistry"),
    ("White Gold","Au-Pd or Au-Ni alloy"),
    ("Rose Gold","Au-Cu alloy with pink hue"),
    ("Sterling Silver","92.5% Ag, 7.5% Cu"),
    ("Britannia Silver","95.8% Ag alloy"),
    ("Electrum","Natural Au-Ag alloy"),
    ("Tungsten Carbide Cemented","WC-Co, extremely hard cutting material"),
    ("Ferrosilicon","Fe-Si alloy, deoxidizer in steelmaking"),
    ("Ferrochrome","Fe-Cr alloy, used to make stainless steel"),
    ("Ferromanganese","Fe-Mn alloy, steel additive"),
]

for name, defn in alloys:
    C(name, defn, "FACT", 0.96)
    R(name, "Alloy", "IS_A", 0.95)

# Link alloys to their base metals
alloy_metals = [
    ("Steel","Iron"),("Stainless Steel","Iron"),("Stainless Steel","Chromium"),
    ("Bronze","Copper"),("Bronze","Tin"),("Brass","Copper"),("Brass","Zinc"),
    ("Duralumin","Aluminium"),("Duralumin","Copper"),
    ("Nichrome","Nickel"),("Nichrome","Chromium"),
    ("Nitinol","Nickel"),("Nitinol","Titanium"),
    ("Sterling Silver","Silver"),("Sterling Silver","Copper"),
    ("White Gold","Gold"),("Rose Gold","Gold"),("Rose Gold","Copper"),
    ("Solder","Tin"),("Solder","Lead"),
    ("Inconel","Nickel"),("Inconel","Chromium"),
    ("Amalgam","Mercury"),
    ("Neodymium Magnet Alloy","Neodymium"),("Neodymium Magnet Alloy","Iron"),
]
for alloy, metal in alloy_metals:
    R(metal, alloy, "PART_OF", 0.9)

# ============================================================
# 11. MINERALS (200+)
# ============================================================
C("Mineral","Naturally occurring inorganic solid with definite chemical composition and crystal structure","DEFINITION",0.99)

minerals = [
    ("Quartz","SiO2, most abundant mineral in Earth's crust, hardness 7"),
    ("Feldspar","Group of aluminosilicate minerals, most abundant mineral group"),
    ("Orthoclase","KAlSi3O8, potassium feldspar, hardness 6"),
    ("Plagioclase","NaAlSi3O8-CaAl2Si2O8, sodium-calcium feldspar series"),
    ("Albite","NaAlSi3O8, sodium plagioclase end-member"),
    ("Anorthite","CaAl2Si2O8, calcium plagioclase end-member"),
    ("Muscovite","KAl2(AlSi3O10)(OH)2, common white mica"),
    ("Biotite","K(Mg,Fe)3AlSi3O10(OH)2, dark mica"),
    ("Phlogopite","KMg3AlSi3O10(OH)2, magnesium mica"),
    ("Lepidolite","K(Li,Al)3(AlSi3O10)(OH)2, lithium mica"),
    ("Chlorite","(Mg,Fe)3(Si,Al)4O10(OH)2·(Mg,Fe)3(OH)6, green sheet silicate"),
    ("Talc","Mg3Si4O10(OH)2, softest mineral (hardness 1)"),
    ("Kaolinite","Al2Si2O5(OH)4, clay mineral for ceramics"),
    ("Montmorillonite","(Na,Ca)0.33(Al,Mg)2Si4O10(OH)2·nH2O, swelling clay"),
    ("Illite","K0.65Al2(Al0.65Si3.35)O10(OH)2, clay mineral"),
    ("Vermiculite","Mg-Fe-Al silicate, expands when heated"),
    ("Serpentine","Mg3Si2O5(OH)4, fibrous or platy mineral"),
    ("Chrysotile","Mg3Si2O5(OH)4, serpentine asbestos"),
    ("Olivine","(Mg,Fe)2SiO4, green mineral in mafic rocks"),
    ("Forsterite","Mg2SiO4, magnesium olivine end-member"),
    ("Fayalite","Fe2SiO4, iron olivine end-member"),
    ("Garnet Group","A3B2(SiO4)3, nesosilicate mineral group"),
    ("Pyrope","Mg3Al2(SiO4)3, red magnesium garnet"),
    ("Almandine","Fe3Al2(SiO4)3, iron-aluminium garnet"),
    ("Spessartine","Mn3Al2(SiO4)3, manganese garnet"),
    ("Grossular","Ca3Al2(SiO4)3, calcium garnet"),
    ("Andradite","Ca3Fe2(SiO4)3, calcium-iron garnet"),
    ("Uvarovite","Ca3Cr2(SiO4)3, rare green chromium garnet"),
    ("Zircon","ZrSiO4, important for radiometric dating"),
    ("Topaz","Al2SiO4(F,OH)2, gemstone, hardness 8"),
    ("Kyanite","Al2SiO5, blue blade-shaped mineral"),
    ("Sillimanite","Al2SiO5, fibrous aluminosilicate polymorph"),
    ("Andalusite","Al2SiO5, prismatic aluminosilicate polymorph"),
    ("Staurolite","Fe2Al9O6(SiO4)4(O,OH)2, cross-shaped twins"),
    ("Epidote","Ca2(Al,Fe)3(SiO4)3(OH), pistachio green mineral"),
    ("Tourmaline","Complex borosilicate, piezoelectric, many colors"),
    ("Beryl","Be3Al2(SiO3)6, gem mineral (emerald, aquamarine)"),
    ("Cordierite","Mg2Al4Si5O18, pleochroic blue mineral"),
    ("Pyroxene Group","Single chain inosilicate minerals"),
    ("Augite","(Ca,Na)(Mg,Fe,Al)(Si,Al)2O6, common pyroxene"),
    ("Diopside","CaMgSi2O6, calcium magnesium pyroxene"),
    ("Enstatite","MgSiO3, magnesium pyroxene"),
    ("Hypersthene","(Mg,Fe)SiO3, orthopyroxene"),
    ("Jadeite","NaAlSi2O6, pyroxene jade"),
    ("Spodumene","LiAlSi2O6, lithium pyroxene, lithium ore"),
    ("Amphibole Group","Double chain inosilicate minerals"),
    ("Hornblende","Complex Ca-Na-Mg-Fe-Al amphibole"),
    ("Actinolite","Ca2(Mg,Fe)5Si8O22(OH)2, green amphibole"),
    ("Tremolite","Ca2Mg5Si8O22(OH)2, white amphibole"),
    ("Glaucophane","Na2Mg3Al2Si8O22(OH)2, blue amphibole in blueschist"),
    ("Calcite","CaCO3, most common carbonate mineral, hardness 3"),
    ("Aragonite","CaCO3, orthorhombic polymorph of calcite"),
    ("Dolomite","CaMg(CO3)2, calcium magnesium carbonate"),
    ("Magnesite","MgCO3, magnesium carbonate mineral"),
    ("Siderite","FeCO3, iron carbonate mineral"),
    ("Rhodochrosite","MnCO3, pink manganese carbonate"),
    ("Smithsonite","ZnCO3, zinc carbonate mineral"),
    ("Cerussite","PbCO3, lead carbonate mineral"),
    ("Malachite","Cu2CO3(OH)2, green copper carbonate, ornamental"),
    ("Azurite","Cu3(CO3)2(OH)2, blue copper carbonate"),
    ("Witherite","BaCO3, barium carbonate mineral"),
    ("Halite","NaCl, rock salt"),
    ("Sylvite","KCl, potassium chloride mineral"),
    ("Fluorite","CaF2, calcium fluoride, hardness 4"),
    ("Cryolite","Na3AlF6, historically used in aluminium smelting"),
    ("Barite","BaSO4, barium sulfate, heavy mineral"),
    ("Celestine","SrSO4, strontium sulfate mineral"),
    ("Anglesite","PbSO4, lead sulfate mineral"),
    ("Anhydrite","CaSO4, anhydrous calcium sulfate"),
    ("Gypsum","CaSO4·2H2O, hydrated calcium sulfate, hardness 2"),
    ("Apatite","Ca5(PO4)3(F,Cl,OH), phosphate mineral, hardness 5"),
    ("Monazite","(Ce,La,Nd,Th)PO4, rare earth phosphate"),
    ("Xenotime","YPO4, yttrium phosphate mineral"),
    ("Turquoise","CuAl6(PO4)4(OH)8·4H2O, blue-green gemstone"),
    ("Vivianite","Fe3(PO4)2·8H2O, blue iron phosphate"),
    ("Pyromorphite","Pb5(PO4)3Cl, lead phosphate mineral"),
    ("Hematite","Fe2O3, most important iron ore"),
    ("Magnetite","Fe3O4, magnetic iron oxide"),
    ("Goethite","FeO(OH), iron oxyhydroxide, brown ore"),
    ("Limonite","FeO(OH)·nH2O, amorphous iron hydroxide"),
    ("Ilmenite","FeTiO3, titanium iron oxide"),
    ("Rutile","TiO2, tetragonal titanium dioxide"),
    ("Anatase","TiO2, metastable titanium dioxide polymorph"),
    ("Brookite","TiO2, orthorhombic titanium dioxide"),
    ("Corundum","Al2O3, hardness 9 (ruby/sapphire when gem-quality)"),
    ("Spinel","MgAl2O4, hard mineral, gemstone"),
    ("Chromite","FeCr2O4, chromium ore"),
    ("Cassiterite","SnO2, tin ore"),
    ("Pyrolusite","MnO2, manganese ore"),
    ("Uraninite","UO2, uranium ore (pitchblende)"),
    ("Bauxite","Al ore mixture (gibbsite, boehmite, diaspore)"),
    ("Gibbsite","Al(OH)3, aluminium hydroxide mineral"),
    ("Boehmite","AlO(OH), aluminium oxyhydroxide"),
    ("Diaspore","AlO(OH), orthorhombic aluminium oxyhydroxide"),
    ("Brucite","Mg(OH)2, magnesium hydroxide mineral"),
    ("Pyrite","FeS2, iron sulfide, fool's gold"),
    ("Marcasite","FeS2, orthorhombic iron sulfide polymorph"),
    ("Chalcopyrite","CuFeS2, most important copper ore"),
    ("Bornite","Cu5FeS4, peacock ore"),
    ("Chalcocite","Cu2S, copper sulfide ore"),
    ("Covellite","CuS, indigo blue copper sulfide"),
    ("Galena","PbS, lead ore, earliest semiconductor"),
    ("Sphalerite","ZnS, zinc ore"),
    ("Wurtzite","ZnS, hexagonal polymorph"),
    ("Cinnabar","HgS, mercury ore, red pigment"),
    ("Stibnite","Sb2S3, antimony ore"),
    ("Molybdenite","MoS2, molybdenum ore, lubricant"),
    ("Arsenopyrite","FeAsS, arsenic iron sulfide"),
    ("Realgar","As4S4, red arsenic sulfide"),
    ("Orpiment","As2S3, yellow arsenic sulfide"),
    ("Pentlandite","(Fe,Ni)9S8, nickel ore"),
    ("Millerite","NiS, nickel sulfide"),
    ("Cobaltite","CoAsS, cobalt ore"),
    ("Skutterudite","CoAs3, cobalt arsenide"),
    ("Sperrylite","PtAs2, platinum arsenide, platinum ore"),
    ("Acanthite","Ag2S, silver sulfide ore"),
    ("Argentite","Ag2S, high-temperature polymorph"),
    ("Proustite","Ag3AsS3, ruby silver"),
    ("Pyrargyrite","Ag3SbS3, dark ruby silver"),
    ("Wolframite","(Fe,Mn)WO4, tungsten ore"),
    ("Scheelite","CaWO4, tungsten ore"),
    ("Columbite","(Fe,Mn)(Nb,Ta)2O6, niobium ore"),
    ("Tantalite","(Fe,Mn)(Ta,Nb)2O6, tantalum ore"),
    ("Bastnasite","(Ce,La)(CO3)F, rare earth ore"),
    ("Xenotime","YPO4, heavy rare earth ore"),
    ("Carnotite","K2(UO2)2(VO4)2·3H2O, uranium-vanadium ore"),
    ("Vanadinite","Pb5(VO4)3Cl, vanadium ore"),
    ("Wulfenite","PbMoO4, lead molybdate"),
    ("Crocoite","PbCrO4, lead chromate, orange mineral"),
    ("Borax","Na2B4O7·10H2O, boron mineral"),
    ("Kernite","Na2B4O7·4H2O, boron mineral"),
    ("Colemanite","Ca2B6O11·5H2O, boron mineral"),
    ("Ulexite","NaCaB5O9·8H2O, TV stone, fiber optic effect"),
    ("Trona","Na3(CO3)(HCO3)·2H2O, sodium sesquicarbonate"),
    ("Natron","Na2CO3·10H2O, hydrated sodium carbonate"),
    ("Mirabilite","Na2SO4·10H2O, Glauber's salt mineral"),
    ("Epsomite","MgSO4·7H2O, Epsom salt mineral"),
    ("Alunite","KAl3(SO4)2(OH)6, potassium aluminium sulfate"),
    ("Jarosite","KFe3(SO4)2(OH)6, iron sulfate mineral (found on Mars)"),
    ("Chalcanthite","CuSO4·5H2O, blue copper sulfate"),
    ("Melanterite","FeSO4·7H2O, green iron sulfate"),
    ("Wollastonite","CaSiO3, calcium silicate"),
    ("Pectolite","NaCa2Si3O8(OH), needle-like silicate"),
    ("Prehnite","Ca2Al2Si3O10(OH)2, pale green mineral"),
    ("Lazurite","(Na,Ca)8(AlSiO4)6(SO4,S,Cl)2, component of lapis lazuli"),
    ("Sodalite","Na8Al6Si6O24Cl2, blue feldspathoid"),
    ("Nepheline","(Na,K)AlSiO4, feldspathoid mineral"),
    ("Leucite","KAlSi2O6, feldspathoid in volcanic rocks"),
    ("Analcime","NaAlSi2O6·H2O, zeolite mineral"),
    ("Natrolite","Na2Al2Si3O10·2H2O, zeolite mineral"),
    ("Stilbite","NaCa4(Si27Al9)O72·28H2O, zeolite mineral"),
    ("Chabazite","(Ca,Na2,K2)Al2Si4O12·6H2O, zeolite"),
    ("Clinoptilolite","(Na,K,Ca)2-3Al3(Al,Si)2Si13O36·12H2O, most abundant zeolite"),
    ("Asbestos","Group of fibrous silicate minerals, carcinogenic"),
    ("Hornfels","Fine-grained metamorphic contact mineral assemblage"),
    ("Garnierite","(Ni,Mg)3Si2O5(OH)4, nickel laterite ore"),
    ("Gibbsite","Al(OH)3, aluminium hydroxide in bauxite"),
    ("Manganite","MnO(OH), manganese oxyhydroxide"),
    ("Todorokite","(Na,Ca,K,Ba,Sr)(Mn,Mg,Al)6O12·3-4H2O, manganese nodule mineral"),
    ("Cuprite","Cu2O, red copper oxide mineral"),
    ("Tenorite","CuO, black copper oxide mineral"),
    ("Zincite","ZnO, zinc oxide mineral"),
    ("Periclase","MgO, magnesium oxide mineral"),
    ("Lime","CaO, calcium oxide mineral"),
    ("Baddeleyite","ZrO2, zirconium dioxide mineral"),
    ("Thorianite","ThO2, thorium dioxide mineral"),
    ("Perovskite","CaTiO3, calcium titanate mineral"),
    ("Stibiconite","Sb3O6(OH), antimony oxide mineral"),
    ("Bismuthinite","Bi2S3, bismuth sulfide mineral"),
    ("Native Gold","Au, gold in native form"),
    ("Native Silver","Ag, silver in native form"),
    ("Native Copper","Cu, copper in native form"),
    ("Native Sulfur","S, sulfur in native form"),
    ("Diamond","C, cubic carbon, hardest natural mineral (10)"),
    ("Graphite","C, hexagonal carbon, soft (1-2), lubricant"),
]

for name, defn in minerals:
    if not any(c["label"]==name for c in concepts):
        C(name, f"Mineral: {defn}", "FACT", 0.96)
    R(name, "Mineral", "IS_A", 0.95)

# ============================================================
# 12. GEMS (55+)
# ============================================================
C("Gemstone","Mineral or organic material cut and polished for jewelry","DEFINITION",0.99)

gems = [
    ("Diamond Gem","Diamond (C), hardness 10, most prized gemstone"),
    ("Ruby","Red corundum (Cr-doped Al2O3), hardness 9"),
    ("Sapphire","Blue corundum (Fe/Ti-doped Al2O3), hardness 9"),
    ("Emerald","Green beryl (Cr/V-doped Be3Al2Si6O18), hardness 7.5-8"),
    ("Aquamarine","Blue beryl (Fe-doped), hardness 7.5-8"),
    ("Morganite","Pink beryl (Mn-doped), hardness 7.5-8"),
    ("Heliodor","Yellow beryl, hardness 7.5-8"),
    ("Alexandrite","Chrysoberyl variety, color-change green to red"),
    ("Cat's Eye Chrysoberyl","Chatoyant chrysoberyl, BeAl2O4"),
    ("Opal","SiO2·nH2O, amorphous silica with play of color"),
    ("Black Opal","Dark-bodied precious opal, most valuable variety"),
    ("Fire Opal","Orange-red opal from Mexico"),
    ("Tanzanite","Blue-purple zoisite from Tanzania"),
    ("Topaz Gem","Al2SiO4(F,OH)2, various colors, hardness 8"),
    ("Imperial Topaz","Orange-pink topaz, most valuable variety"),
    ("Amethyst","Purple quartz (Fe-irradiated), hardness 7"),
    ("Citrine","Yellow-orange quartz, hardness 7"),
    ("Rose Quartz","Pink quartz, hardness 7"),
    ("Smoky Quartz","Brown-gray quartz, hardness 7"),
    ("Tiger's Eye","Chatoyant quartz with golden-brown bands"),
    ("Aventurine","Quartz with sparkly inclusions"),
    ("Chalcedony","Microcrystalline quartz group"),
    ("Agate","Banded chalcedony in various colors"),
    ("Carnelian","Orange-red chalcedony"),
    ("Onyx","Black and white banded chalcedony"),
    ("Jasper","Opaque microcrystalline quartz, many patterns"),
    ("Bloodstone","Green jasper with red spots"),
    ("Chrysoprase","Apple-green chalcedony (Ni-colored)"),
    ("Tourmaline Gem","Complex borosilicate in many colors"),
    ("Rubellite","Red-pink tourmaline"),
    ("Indicolite","Blue tourmaline"),
    ("Paraiba Tourmaline","Neon blue-green tourmaline (Cu-colored)"),
    ("Watermelon Tourmaline","Pink center, green rim tourmaline"),
    ("Garnet Gem","Garnet group gemstones in various colors"),
    ("Tsavorite","Green grossular garnet (V/Cr-colored)"),
    ("Demantoid","Green andradite garnet, high dispersion"),
    ("Rhodolite","Purple-red pyrope-almandine garnet"),
    ("Spinel Gem","MgAl2O4, various colors, historically confused with ruby"),
    ("Peridot","Gem olivine (Mg2SiO4), yellow-green"),
    ("Zircon Gem","ZrSiO4, high brilliance, various colors"),
    ("Lapis Lazuli","Blue metamorphic rock (lazurite + pyrite + calcite)"),
    ("Turquoise Gem","CuAl6(PO4)4(OH)8·4H2O, blue-green gem"),
    ("Jade Nephrite","Ca2(Mg,Fe)5Si8O22(OH)2, tough green gem"),
    ("Jade Jadeite","NaAlSi2O6, more valuable jade variety"),
    ("Moonstone","Orthoclase feldspar with adularescence"),
    ("Labradorite","Plagioclase feldspar with labradorescence"),
    ("Sunstone","Feldspar with aventurescent inclusions"),
    ("Pearl","Organic gem (CaCO3 nacre from mollusks)"),
    ("Amber","Fossilized tree resin, organic gem"),
    ("Coral Gem","Organic gem from coral skeletons (CaCO3)"),
    ("Jet","Fossilized wood, black organic gem"),
    ("Ivory","Organic gem from elephant tusks (dentine)"),
    ("Kunzite","Pink spodumene gem (Mn-colored)"),
    ("Hiddenite","Green spodumene gem (Cr-colored)"),
    ("Sphene","CaTiSiO5, high dispersion gem (titanite)"),
    ("Benitoite","BaTiSi3O9, rare blue gem from California"),
    ("Tanzanite","Blue-violet zoisite, Ca2Al3(SiO4)3(OH)"),
    ("Iolite","Gem cordierite, violet-blue pleochroic"),
    ("Dioptase","CuSiO3·H2O, vivid emerald-green"),
    ("Sugilite","KNa2(Fe,Mn,Al)2Li3Si12O30, purple gem"),
    ("Charoite","K(Ca,Na)2Si4O10(OH,F)·H2O, purple Siberian gem"),
    ("Larimar","Blue pectolite from Dominican Republic"),
    ("Rhodonite","MnSiO3, pink manganese silicate gem"),
    ("Malachite Gem","Cu2CO3(OH)2, banded green ornamental gem"),
]

for name, defn in gems:
    C(name, f"Gemstone: {defn}", "FACT", 0.96)
    R(name, "Gemstone", "IS_A", 0.95)

gem_mineral_links = [
    ("Ruby","Corundum"),("Sapphire","Corundum"),
    ("Emerald","Beryl"),("Aquamarine","Beryl"),("Morganite","Beryl"),
    ("Amethyst","Quartz"),("Citrine","Quartz"),("Rose Quartz","Quartz"),
    ("Peridot","Olivine"),("Zircon Gem","Zircon"),
    ("Spinel Gem","Spinel"),("Pearl","Calcium Carbonate"),
    ("Opal","Silicon Dioxide"),("Diamond Gem","Diamond"),
    ("Lapis Lazuli","Lazurite"),("Turquoise Gem","Turquoise"),
    ("Jade Jadeite","Jadeite"),("Moonstone","Orthoclase"),
    ("Kunzite","Spodumene"),("Hiddenite","Spodumene"),
    ("Topaz Gem","Topaz"),("Tourmaline Gem","Tourmaline"),
]
for gem, mineral in gem_mineral_links:
    R(gem, mineral, "RELATES_TO", 0.9)

# Chemistry concepts & laws
chem_concepts = [
    ("Mole","SI unit of amount of substance, 6.022e23 entities","DEFINITION"),
    ("Avogadro's Number","6.022e23, number of entities in one mole","FACT"),
    ("Molar Mass","Mass of one mole of a substance in g/mol","DEFINITION"),
    ("Molarity","Concentration in moles of solute per liter of solution","DEFINITION"),
    ("pH","Negative logarithm of hydrogen ion concentration","DEFINITION"),
    ("pKa","Negative logarithm of acid dissociation constant Ka","DEFINITION"),
    ("Electronegativity","Tendency of atom to attract shared electrons","DEFINITION"),
    ("Ionization Energy","Energy to remove electron from gaseous atom","DEFINITION"),
    ("Covalent Bond","Chemical bond by sharing electron pairs","DEFINITION"),
    ("Ionic Bond","Chemical bond by electrostatic attraction of ions","DEFINITION"),
    ("Metallic Bond","Delocalized electron sea bonding in metals","DEFINITION"),
    ("Hydrogen Bond","Strong intermolecular force between H and F/O/N","DEFINITION"),
    ("Van der Waals Forces","Weak intermolecular forces (London dispersion)","DEFINITION"),
    ("VSEPR Theory","Predicts molecular geometry from electron pair repulsion","THEORY"),
    ("Hybridization","Mixing of atomic orbitals to form hybrid orbitals","THEORY"),
    ("Le Chatelier's Principle","System at equilibrium shifts to counteract disturbance","THEORY"),
    ("Hess's Law","Total enthalpy change is independent of pathway","THEORY"),
    ("Ideal Gas Law","PV = nRT","THEORY"),
    ("Arrhenius Equation","k = A*exp(-Ea/RT), rate constant temperature dependence","THEORY"),
    ("Nernst Equation","Relates electrode potential to concentrations","THEORY"),
    ("Gibbs Free Energy","G = H - TS, criterion for spontaneity","DEFINITION"),
    ("Enthalpy","H, heat content at constant pressure","DEFINITION"),
    ("Entropy","S, measure of disorder/microstates","DEFINITION"),
    ("Activation Energy","Minimum energy required for reaction to occur","DEFINITION"),
    ("Equilibrium Constant","Ratio of products to reactants at equilibrium","DEFINITION"),
    ("Oxidation State","Hypothetical charge if all bonds were ionic","DEFINITION"),
    ("Crystal Field Theory","Explains colors and magnetism of transition metal complexes","THEORY"),
    ("Chirality","Non-superimposable mirror image property","DEFINITION"),
    ("Aromaticity","Cyclic conjugation with 4n+2 pi electrons (Huckel rule)","THEORY"),
    ("Resonance","Delocalization of electrons described by contributing structures","THEORY"),
    ("Markovnikov's Rule","H adds to less substituted carbon of alkene","THEORY"),
    ("Buffer Solution","Solution resisting pH change upon acid/base addition","DEFINITION"),
    ("Titration","Quantitative analysis by adding measured reagent to analyte","DEFINITION"),
    ("NMR Spectroscopy","Nuclear magnetic resonance, determines molecular structure","DEFINITION"),
    ("IR Spectroscopy","Infrared spectroscopy, identifies functional groups","DEFINITION"),
    ("Mass Spectrometry","Measures mass-to-charge ratio of ions","DEFINITION"),
    ("Chromatography","Separation technique based on differential partitioning","DEFINITION"),
    ("HPLC","High-performance liquid chromatography","DEFINITION"),
    ("X-ray Crystallography","Structure determination by X-ray diffraction","DEFINITION"),
    ("Colligative Properties","Properties depending on solute particle count","DEFINITION"),
    ("Woodward-Hoffmann Rules","Orbital symmetry controls pericyclic reactions","THEORY"),
]

for label, defn, typ in chem_concepts:
    C(label, defn, typ, 0.97)

for tech in ["NMR Spectroscopy","IR Spectroscopy","Mass Spectrometry","HPLC","X-ray Crystallography"]:
    R(tech, "Analytical Chemistry", "PART_OF", 0.9)

# Cross-relations
R("Haber Process","Ammonia","ENABLES",0.95)
R("Haber Process","Iron Catalyst","USED_IN",0.9)
R("Contact Process","Sulfuric Acid","ENABLES",0.95)
R("Ostwald Process","Nitric Acid","ENABLES",0.95)
R("Fermentation","Ethanol","ENABLES",0.9)
R("Photosynthesis","Glucose","ENABLES",0.95)
R("Photosynthesis","Chlorophyll","USED_IN",0.95)
R("Polymerization","Polymer","ENABLES",0.95)

mineral_elem = [
    ("Hematite","Iron"),("Magnetite","Iron"),("Chalcopyrite","Copper"),
    ("Galena","Lead"),("Sphalerite","Zinc"),("Cassiterite","Tin"),
    ("Bauxite","Aluminium"),("Chromite","Chromium"),("Uraninite","Uranium"),
    ("Cinnabar","Mercury"),("Scheelite","Tungsten"),("Pentlandite","Nickel"),
    ("Molybdenite","Molybdenum"),("Spodumene","Lithium"),("Monazite","Cerium"),
]
for mineral, elem in mineral_elem:
    R(mineral, elem, "ENABLES", 0.85)

C("Amino Acid","Building block of proteins, contains amino and carboxyl groups","DEFINITION",0.99)
amino_acids = ["Glycine","Alanine","Valine","Leucine","Isoleucine","Proline",
    "Phenylalanine","Tryptophan","Methionine","Serine","Threonine","Cysteine",
    "Tyrosine","Asparagine","Glutamine","Aspartic Acid","Glutamic Acid",
    "Lysine","Arginine","Histidine"]
for aa in amino_acids:
    R(aa, "Amino Acid", "IS_A", 0.95)
    R(aa, "Biochemistry", "PART_OF", 0.85)

for nb in ["Adenine","Guanine","Cytosine","Thymine","Uracil"]:
    R(nb, "DNA", "PART_OF", 0.9)

# OUTPUT
data = {"concepts": concepts, "relations": relations}
print(f"Concepts: {len(concepts)}")
print(f"Relations: {len(relations)}")

with open("/home/hirschpekf/brain19/data/wave19_chemistry.json", "w") as f:
    json.dump(data, f, indent=1)
print("Written to wave19_chemistry.json")
