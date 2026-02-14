#!/usr/bin/env python3
"""Generate Wave 13: Medicine + Biology Deep knowledge base."""
import json

concepts = []
relations = []

def C(label, definition, typ="FACT", trust=0.95):
    concepts.append({"label": label, "definition": definition, "type": typ, "trust": trust})

def R(src, tgt, typ="RELATES_TO", w=0.9):
    relations.append({"source": src, "target": tgt, "type": typ, "weight": w})

# ============================================================
# AMINO ACIDS (20 standard + extras)
# ============================================================
amino_acids = {
    "Glycine": ("simplest amino acid, single hydrogen side chain, inhibitory neurotransmitter in CNS", 0.98),
    "Alanine": ("nonpolar amino acid with methyl side chain, major substrate for gluconeogenesis", 0.97),
    "Valine": ("branched-chain essential amino acid, important for muscle metabolism", 0.97),
    "Leucine": ("branched-chain essential amino acid, activates mTOR pathway for protein synthesis", 0.97),
    "Isoleucine": ("branched-chain essential amino acid, involved in hemoglobin synthesis and energy regulation", 0.97),
    "Proline": ("cyclic amino acid, important for collagen structure, introduces kinks in polypeptide chains", 0.97),
    "Phenylalanine": ("aromatic essential amino acid, precursor to tyrosine, accumulated in phenylketonuria", 0.97),
    "Tryptophan": ("aromatic essential amino acid, precursor to serotonin and melatonin, contains indole ring", 0.97),
    "Methionine": ("sulfur-containing essential amino acid, start codon amino acid, methyl group donor as SAM", 0.97),
    "Serine": ("polar amino acid, phosphorylation site in signaling, precursor to glycine and cysteine", 0.97),
    "Threonine": ("polar essential amino acid, has hydroxyl group, O-linked glycosylation site", 0.97),
    "Cysteine": ("sulfur-containing amino acid, forms disulfide bonds, part of glutathione", 0.97),
    "Tyrosine": ("aromatic amino acid, precursor to catecholamines and thyroid hormones, phosphorylation site", 0.97),
    "Asparagine": ("polar amino acid, N-linked glycosylation site, product of transamination of oxaloacetate", 0.96),
    "Glutamine": ("most abundant free amino acid in blood, nitrogen carrier, fuel for enterocytes and immune cells", 0.97),
    "Aspartate": ("acidic amino acid, excitatory neurotransmitter, participates in urea cycle and malate-aspartate shuttle", 0.97),
    "Glutamate": ("acidic amino acid, major excitatory neurotransmitter in brain, umami taste", 0.97),
    "Lysine": ("basic essential amino acid, important for collagen cross-linking and carnitine synthesis", 0.97),
    "Arginine": ("basic amino acid, precursor to nitric oxide, involved in urea cycle", 0.97),
    "Histidine": ("basic essential amino acid, precursor to histamine, contains imidazole ring", 0.97),
}
essential = ["Valine","Leucine","Isoleucine","Phenylalanine","Tryptophan","Methionine","Threonine","Lysine","Histidine"]
for aa, (defn, t) in amino_acids.items():
    C(aa, defn, "DEFINITION", t)
    R(aa, "Amino Acid", "IS_A", 0.99)
    R(aa, "Protein Synthesis", "USED_IN", 0.95)
    if aa in essential:
        R(aa, "Essential Amino Acid", "IS_A", 0.99)

C("Amino Acid", "organic molecule containing amino and carboxyl groups, building blocks of proteins", "DEFINITION", 0.99)
C("Essential Amino Acid", "amino acid that cannot be synthesized by the body and must be obtained from diet", "DEFINITION", 0.99)
C("Branched-Chain Amino Acid", "leucine, isoleucine, valine - metabolized in muscle rather than liver", "DEFINITION", 0.97)
for bcaa in ["Valine","Leucine","Isoleucine"]:
    R(bcaa, "Branched-Chain Amino Acid", "IS_A", 0.99)
R("Phenylalanine", "Tyrosine", "ENABLES", 0.95)
R("Tryptophan", "Serotonin", "ENABLES", 0.95)
R("Tryptophan", "Melatonin", "ENABLES", 0.93)
R("Tyrosine", "Dopamine", "ENABLES", 0.95)
R("Tyrosine", "Norepinephrine", "ENABLES", 0.93)
R("Tyrosine", "Thyroid Hormones", "ENABLES", 0.92)
R("Histidine", "Histamine", "ENABLES", 0.95)
R("Arginine", "Nitric Oxide", "ENABLES", 0.95)
R("Glutamate", "GABA", "ENABLES", 0.93)
R("Methionine", "S-Adenosylmethionine", "ENABLES", 0.94)
R("Cysteine", "Glutathione", "PART_OF", 0.95)

# ============================================================
# VITAMINS (13)
# ============================================================
vitamins = {
    "Vitamin A": ("fat-soluble vitamin (retinol/retinal/retinoic acid), essential for vision, immune function, cell differentiation", "Retinol", ["Night Blindness","Xerophthalmia","Bitot Spots"]),
    "Vitamin B1": ("thiamine, water-soluble, coenzyme in pyruvate dehydrogenase and alpha-ketoglutarate dehydrogenase", "Thiamine", ["Beriberi","Wernicke Encephalopathy","Korsakoff Syndrome"]),
    "Vitamin B2": ("riboflavin, component of FAD and FMN coenzymes, involved in electron transport chain", "Riboflavin", ["Ariboflavinosis","Angular Cheilitis"]),
    "Vitamin B3": ("niacin, component of NAD+ and NADP+, involved in redox reactions and DNA repair", "Niacin", ["Pellagra"]),
    "Vitamin B5": ("pantothenic acid, component of coenzyme A, essential for fatty acid synthesis and oxidation", "Pantothenic Acid", ["Dermatitis","Paresthesia"]),
    "Vitamin B6": ("pyridoxine/pyridoxal/pyridoxamine, coenzyme in amino acid metabolism and neurotransmitter synthesis", "Pyridoxine", ["Peripheral Neuropathy","Sideroblastic Anemia"]),
    "Vitamin B7": ("biotin, coenzyme for carboxylase enzymes, important in gluconeogenesis and fatty acid synthesis", "Biotin", ["Dermatitis","Alopecia"]),
    "Vitamin B9": ("folate/folic acid, essential for one-carbon metabolism, DNA synthesis, and neural tube development", "Folate", ["Megaloblastic Anemia","Neural Tube Defects"]),
    "Vitamin B12": ("cobalamin, required for methionine synthase and methylmalonyl-CoA mutase, needs intrinsic factor for absorption", "Cobalamin", ["Pernicious Anemia","Subacute Combined Degeneration"]),
    "Vitamin C": ("ascorbic acid, water-soluble antioxidant, cofactor for collagen synthesis (prolyl/lysyl hydroxylase), enhances iron absorption", "Ascorbic Acid", ["Scurvy"]),
    "Vitamin D": ("fat-soluble secosteroid, synthesized in skin via UV, regulates calcium/phosphate homeostasis", "Cholecalciferol", ["Rickets","Osteomalacia"]),
    "Vitamin E": ("tocopherol, fat-soluble antioxidant protecting cell membranes from lipid peroxidation", "Tocopherol", ["Hemolytic Anemia","Ataxia"]),
    "Vitamin K": ("fat-soluble vitamin, cofactor for gamma-carboxylation of clotting factors II VII IX X and proteins C and S", "Phylloquinone", ["Hemorrhagic Disease of Newborn","Coagulopathy"]),
}
for v, (defn, alt, deficiencies) in vitamins.items():
    C(v, defn, "DEFINITION", 0.97)
    R(v, "Vitamin", "IS_A", 0.99)
    for d in deficiencies:
        if d not in [c["label"] for c in concepts]:
            C(d, f"disease caused by deficiency of {v}", "FACT", 0.95)
        R(v+" Deficiency", d, "CAUSES", 0.93)
        C(v+" Deficiency", f"insufficient levels of {v} in the body", "FACT", 0.95)
        R(v, v+" Deficiency", "INHIBITS", 0.9)

C("Vitamin", "essential organic micronutrient required in small amounts for normal metabolism", "DEFINITION", 0.99)

# ============================================================
# HORMONES (60+)
# ============================================================
hormones = {
    # Hypothalamic
    "Gonadotropin-Releasing Hormone": ("hypothalamic peptide hormone stimulating FSH and LH release from anterior pituitary", "Hypothalamus", ["FSH","LH"]),
    "Thyrotropin-Releasing Hormone": ("hypothalamic tripeptide stimulating TSH and prolactin release", "Hypothalamus", ["TSH"]),
    "Corticotropin-Releasing Hormone": ("hypothalamic peptide stimulating ACTH release, key stress response mediator", "Hypothalamus", ["ACTH"]),
    "Growth Hormone-Releasing Hormone": ("hypothalamic peptide stimulating growth hormone secretion", "Hypothalamus", ["Growth Hormone"]),
    "Somatostatin": ("inhibitory peptide hormone, inhibits GH, TSH, insulin, glucagon release", "Hypothalamus", []),
    "Dopamine (Hormone)": ("hypothalamic catecholamine that inhibits prolactin secretion", "Hypothalamus", []),
    # Anterior Pituitary
    "Growth Hormone": ("anterior pituitary peptide promoting growth via IGF-1, increases blood glucose", "Anterior Pituitary", ["IGF-1"]),
    "TSH": ("thyroid-stimulating hormone, glycoprotein stimulating thyroid hormone synthesis and release", "Anterior Pituitary", ["T3","T4"]),
    "ACTH": ("adrenocorticotropic hormone, stimulates cortisol production from adrenal cortex", "Anterior Pituitary", ["Cortisol"]),
    "FSH": ("follicle-stimulating hormone, stimulates follicular development and spermatogenesis", "Anterior Pituitary", []),
    "LH": ("luteinizing hormone, triggers ovulation and testosterone production", "Anterior Pituitary", []),
    "Prolactin": ("anterior pituitary hormone promoting lactation and breast development", "Anterior Pituitary", []),
    # Posterior Pituitary
    "Oxytocin": ("posterior pituitary peptide causing uterine contractions and milk ejection, promotes bonding", "Posterior Pituitary", []),
    "Vasopressin": ("antidiuretic hormone (ADH), promotes water reabsorption in collecting ducts via aquaporin-2", "Posterior Pituitary", []),
    # Thyroid
    "T3": ("triiodothyronine, active thyroid hormone regulating metabolism, produced by deiodination of T4", "Thyroid Gland", []),
    "T4": ("thyroxine, primary thyroid hormone secretion, converted to active T3 peripherally", "Thyroid Gland", []),
    "Calcitonin": ("thyroid C-cell peptide that lowers blood calcium by inhibiting osteoclasts", "Thyroid Gland", []),
    # Parathyroid
    "Parathyroid Hormone": ("PTH, increases blood calcium via bone resorption, renal reabsorption, and vitamin D activation", "Parathyroid Gland", []),
    # Adrenal Cortex
    "Cortisol": ("glucocorticoid from zona fasciculata, anti-inflammatory, increases blood glucose, stress hormone", "Adrenal Cortex", []),
    "Aldosterone": ("mineralocorticoid from zona glomerulosa, promotes sodium retention and potassium excretion in kidneys", "Adrenal Cortex", []),
    "DHEA": ("dehydroepiandrosterone, adrenal androgen precursor from zona reticularis", "Adrenal Cortex", []),
    # Adrenal Medulla
    "Epinephrine": ("catecholamine from adrenal medulla, fight-or-flight response, increases HR and blood glucose", "Adrenal Medulla", []),
    "Norepinephrine (Hormone)": ("catecholamine from adrenal medulla, vasoconstriction, increases blood pressure", "Adrenal Medulla", []),
    # Pancreas
    "Insulin": ("pancreatic beta-cell peptide hormone, lowers blood glucose by promoting cellular uptake and glycogen synthesis", "Pancreatic Beta Cell", []),
    "Glucagon": ("pancreatic alpha-cell peptide, raises blood glucose via glycogenolysis and gluconeogenesis", "Pancreatic Alpha Cell", []),
    "Somatostatin (Pancreatic)": ("pancreatic delta-cell hormone inhibiting insulin and glucagon secretion", "Pancreatic Delta Cell", []),
    "Pancreatic Polypeptide": ("F-cell hormone regulating pancreatic secretion and gastrointestinal motility", "Pancreatic F Cell", []),
    # Gonads
    "Testosterone": ("primary androgen, promotes male secondary sexual characteristics, spermatogenesis, muscle mass", "Testes", []),
    "Estradiol": ("primary estrogen, promotes female secondary sexual characteristics, endometrial growth, bone density", "Ovary", []),
    "Progesterone": ("steroid hormone maintaining pregnancy, promotes secretory endometrium, raises body temperature", "Ovary", []),
    "Inhibin": ("gonadal glycoprotein selectively inhibiting FSH secretion from anterior pituitary", "Gonads", []),
    "Activin": ("gonadal glycoprotein stimulating FSH secretion, opposite of inhibin", "Gonads", []),
    "Anti-Mullerian Hormone": ("glycoprotein from Sertoli/granulosa cells, inhibits Mullerian duct development, ovarian reserve marker", "Gonads", []),
    # GI Hormones
    "Gastrin": ("G-cell peptide stimulating gastric acid (HCl) secretion and gastric mucosal growth", "G Cells (Stomach)", []),
    "Secretin": ("S-cell hormone stimulating bicarbonate secretion from pancreas, inhibits gastric acid", "S Cells (Duodenum)", []),
    "Cholecystokinin": ("I-cell hormone stimulating gallbladder contraction and pancreatic enzyme secretion, satiety signal", "I Cells (Duodenum)", []),
    "Ghrelin": ("stomach fundus hormone stimulating appetite and growth hormone release, hunger hormone", "Stomach", []),
    "Leptin": ("adipocyte hormone signaling satiety, inhibits appetite via hypothalamus", "Adipose Tissue", []),
    "Adiponectin": ("adipokine enhancing insulin sensitivity and fatty acid oxidation, anti-inflammatory", "Adipose Tissue", []),
    "GLP-1": ("glucagon-like peptide-1, incretin hormone enhancing insulin secretion, slows gastric emptying", "L Cells (Ileum)", []),
    "GIP": ("glucose-dependent insulinotropic peptide, incretin from K cells enhancing insulin secretion", "K Cells (Duodenum)", []),
    "Motilin": ("peptide hormone initiating migrating motor complex in fasting state", "M Cells (Duodenum)", []),
    "VIP": ("vasoactive intestinal peptide, relaxes smooth muscle, stimulates water/electrolyte secretion", "Enteric Neurons", []),
    # Kidney
    "Erythropoietin": ("glycoprotein hormone from kidney stimulating red blood cell production in bone marrow", "Kidney", []),
    "Renin": ("aspartyl protease from juxtaglomerular cells, cleaves angiotensinogen to angiotensin I", "Kidney", []),
    "Calcitriol": ("active vitamin D (1,25-dihydroxycholecalciferol), promotes calcium absorption in gut", "Kidney", []),
    # Heart
    "ANP": ("atrial natriuretic peptide, released by atrial stretch, promotes sodium excretion and vasodilation", "Heart Atria", []),
    "BNP": ("brain natriuretic peptide, released by ventricles in heart failure, promotes natriuresis", "Heart Ventricles", []),
    # Pineal
    "Melatonin": ("pineal gland hormone regulating circadian rhythm, synthesized from serotonin in darkness", "Pineal Gland", []),
    # Thymus
    "Thymosin": ("thymic peptide promoting T-cell maturation and differentiation", "Thymus", []),
    # Placenta
    "hCG": ("human chorionic gonadotropin, maintains corpus luteum in early pregnancy, basis of pregnancy tests", "Placenta", []),
    "Human Placental Lactogen": ("placental hormone with anti-insulin effects, promotes fetal growth", "Placenta", []),
    # Other
    "IGF-1": ("insulin-like growth factor 1, mediates growth hormone effects on tissues, produced mainly in liver", "Liver", []),
    "Angiotensin II": ("potent vasoconstrictor peptide, stimulates aldosterone release, part of RAAS", "Systemic", []),
    "Thrombopoietin": ("liver-produced hormone stimulating platelet production from megakaryocytes", "Liver", []),
    "Hepcidin": ("liver peptide hormone, master regulator of iron homeostasis, degrades ferroportin", "Liver", []),
}

for h, (defn, source, targets) in hormones.items():
    C(h, defn, "DEFINITION", 0.96)
    R(h, "Hormone", "IS_A", 0.98)
    R(source, h, "ENABLES", 0.92)
    for t in targets:
        R(h, t, "ENABLES", 0.9)

C("Hormone", "chemical messenger produced by endocrine glands or tissues, transported via blood to target organs", "DEFINITION", 0.99)

# Key hormone relationships
R("Insulin", "Blood Glucose", "INHIBITS", 0.97)
R("Glucagon", "Blood Glucose", "ENABLES", 0.95)
R("Cortisol", "Blood Glucose", "ENABLES", 0.9)
R("Cortisol", "Immune Response", "INHIBITS", 0.9)
R("Somatostatin", "Growth Hormone", "INHIBITS", 0.93)
R("Somatostatin", "Insulin", "INHIBITS", 0.9)
R("Somatostatin", "Glucagon", "INHIBITS", 0.9)
R("Dopamine (Hormone)", "Prolactin", "INHIBITS", 0.95)
R("Inhibin", "FSH", "INHIBITS", 0.95)
R("Leptin", "Appetite", "INHIBITS", 0.93)
R("Ghrelin", "Appetite", "ENABLES", 0.93)
R("Calcitonin", "Blood Calcium", "INHIBITS", 0.93)
R("Parathyroid Hormone", "Blood Calcium", "ENABLES", 0.95)
R("Aldosterone", "Sodium Reabsorption", "ENABLES", 0.95)
R("Aldosterone", "Potassium Excretion", "ENABLES", 0.93)
R("Vasopressin", "Water Reabsorption", "ENABLES", 0.95)
R("ANP", "Sodium Excretion", "ENABLES", 0.92)
R("ANP", "Blood Pressure", "INHIBITS", 0.9)
R("Renin", "Angiotensin II", "ENABLES", 0.95)
R("Angiotensin II", "Aldosterone", "ENABLES", 0.93)
R("Angiotensin II", "Blood Pressure", "ENABLES", 0.95)
R("Hepcidin", "Iron Absorption", "INHIBITS", 0.93)
R("Erythropoietin", "Red Blood Cell Production", "ENABLES", 0.96)

# ============================================================
# NEUROTRANSMITTERS (25+)
# ============================================================
neurotransmitters = {
    "Dopamine": ("catecholamine neurotransmitter involved in reward, motivation, motor control; deficit in Parkinson's", "DEFINITION"),
    "Serotonin": ("5-HT, monoamine neurotransmitter regulating mood, sleep, appetite; target of SSRIs", "DEFINITION"),
    "Norepinephrine": ("catecholamine neurotransmitter for alertness, attention, fight-or-flight in CNS and PNS", "DEFINITION"),
    "Acetylcholine": ("neurotransmitter at neuromuscular junction and parasympathetic system, involved in memory", "DEFINITION"),
    "GABA": ("gamma-aminobutyric acid, primary inhibitory neurotransmitter in CNS, opens chloride channels", "DEFINITION"),
    "Glutamate (NT)": ("primary excitatory neurotransmitter in CNS, acts on NMDA/AMPA/kainate receptors", "DEFINITION"),
    "Glycine (NT)": ("inhibitory neurotransmitter in spinal cord and brainstem, co-agonist at NMDA receptors", "DEFINITION"),
    "Histamine (NT)": ("monoamine neurotransmitter regulating wakefulness, gastric acid secretion, immune response", "DEFINITION"),
    "Endorphin": ("endogenous opioid neuropeptide, pain modulation and reward, released during exercise and stress", "DEFINITION"),
    "Enkephalin": ("endogenous opioid pentapeptide, pain modulation, found in CNS and adrenal medulla", "DEFINITION"),
    "Dynorphin": ("endogenous opioid peptide acting on kappa receptors, involved in pain and stress response", "DEFINITION"),
    "Substance P": ("neuropeptide involved in pain transmission, inflammation, and nausea/vomiting", "DEFINITION"),
    "Neuropeptide Y": ("abundant CNS peptide stimulating appetite, reducing anxiety, vasoconstriction", "DEFINITION"),
    "Nitric Oxide (NT)": ("gaseous neurotransmitter causing vasodilation, retrograde signaling in synapses", "DEFINITION"),
    "Adenosine": ("purine nucleoside neuromodulator promoting sleep and inhibiting arousal, antagonized by caffeine", "DEFINITION"),
    "ATP (NT)": ("purinergic neurotransmitter, co-released with other transmitters, acts on P2 receptors", "DEFINITION"),
    "Anandamide": ("endocannabinoid neurotransmitter, activates CB1 receptors, modulates pain and mood", "DEFINITION"),
    "2-AG": ("2-arachidonoylglycerol, most abundant endocannabinoid, retrograde synaptic messenger", "DEFINITION"),
    "Oxytocin (NT)": ("neuropeptide promoting social bonding, trust, and pair bonding in brain circuits", "DEFINITION"),
    "Vasopressin (NT)": ("neuropeptide modulating social behavior, aggression, and pair bonding", "DEFINITION"),
    "CGRP": ("calcitonin gene-related peptide, potent vasodilator, key mediator in migraine pathophysiology", "DEFINITION"),
    "Orexin": ("hypothalamic neuropeptide promoting wakefulness and appetite, deficient in narcolepsy", "DEFINITION"),
    "D-Serine": ("co-agonist at NMDA receptors, important for synaptic plasticity and memory", "DEFINITION"),
    "Agmatine": ("decarboxylated arginine, neuromodulator with NMDA antagonist and imidazoline agonist properties", "DEFINITION"),
}
for nt, (defn, typ) in neurotransmitters.items():
    C(nt, defn, typ, 0.96)
    R(nt, "Neurotransmitter", "IS_A", 0.98)

C("Neurotransmitter", "chemical messenger transmitting signals across synapses between neurons", "DEFINITION", 0.99)
R("Caffeine", "Adenosine", "INHIBITS", 0.95)
C("Caffeine", "methylxanthine stimulant blocking adenosine receptors, increasing alertness", "DEFINITION", 0.97)

# ============================================================
# ORGAN SYSTEMS (11 detailed)
# ============================================================
organ_systems = {
    "Cardiovascular System": ("organ system consisting of heart, blood vessels, and blood, responsible for circulation", 
        ["Heart","Aorta","Superior Vena Cava","Inferior Vena Cava","Pulmonary Artery","Pulmonary Vein","Coronary Arteries",
         "Left Ventricle","Right Ventricle","Left Atrium","Right Atrium","Mitral Valve","Aortic Valve","Tricuspid Valve",
         "Pulmonary Valve","Sinoatrial Node","Atrioventricular Node","Bundle of His","Purkinje Fibers"]),
    "Respiratory System": ("organ system for gas exchange, bringing oxygen into body and removing carbon dioxide",
        ["Lungs","Trachea","Bronchi","Bronchioles","Alveoli","Diaphragm","Pleura","Larynx","Pharynx","Nasal Cavity",
         "Epiglottis","Right Lung","Left Lung"]),
    "Nervous System": ("organ system controlling body functions via electrical and chemical signals",
        ["Brain","Spinal Cord","Cerebrum","Cerebellum","Brainstem","Medulla Oblongata","Pons","Midbrain","Thalamus",
         "Hypothalamus","Hippocampus","Amygdala","Basal Ganglia","Corpus Callosum","Frontal Lobe","Parietal Lobe",
         "Temporal Lobe","Occipital Lobe","Peripheral Nerves","Sympathetic Nervous System","Parasympathetic Nervous System"]),
    "Digestive System": ("organ system breaking down food into nutrients for absorption and eliminating waste",
        ["Mouth","Esophagus","Stomach","Small Intestine","Duodenum","Jejunum","Ileum","Large Intestine","Cecum",
         "Appendix","Colon","Rectum","Anus","Liver","Gallbladder","Pancreas","Salivary Glands","Tongue"]),
    "Musculoskeletal System": ("organ system providing structure, support, movement, and protection",
        ["Skeletal Muscle","Smooth Muscle","Cardiac Muscle","Bone","Cartilage","Tendon","Ligament","Joint",
         "Femur","Tibia","Humerus","Radius","Ulna","Vertebral Column","Pelvis","Skull","Ribcage","Scapula"]),
    "Endocrine System": ("organ system of glands producing hormones regulating metabolism, growth, and reproduction",
        ["Hypothalamus","Pituitary Gland","Anterior Pituitary","Posterior Pituitary","Thyroid Gland","Parathyroid Gland",
         "Adrenal Gland","Adrenal Cortex","Adrenal Medulla","Pancreatic Islets","Pineal Gland","Thymus","Ovary","Testes"]),
    "Urinary System": ("organ system filtering blood to produce urine, maintaining fluid and electrolyte balance",
        ["Kidney","Renal Cortex","Renal Medulla","Nephron","Glomerulus","Bowman Capsule","Proximal Convoluted Tubule",
         "Loop of Henle","Distal Convoluted Tubule","Collecting Duct","Ureter","Urinary Bladder","Urethra"]),
    "Immune System": ("defense system protecting against pathogens via innate and adaptive immunity",
        ["Bone Marrow","Thymus","Spleen","Lymph Nodes","Tonsils","MALT","Peyer Patches"]),
    "Reproductive System": ("organ system for producing offspring",
        ["Testes","Epididymis","Vas Deferens","Seminal Vesicle","Prostate","Ovary","Fallopian Tube","Uterus","Cervix","Vagina"]),
    "Integumentary System": ("organ system consisting of skin, hair, nails, and associated glands",
        ["Epidermis","Dermis","Hypodermis","Hair Follicle","Sebaceous Gland","Sweat Gland","Nail"]),
    "Lymphatic System": ("network of vessels and organs draining interstitial fluid and transporting immune cells",
        ["Lymph Vessels","Thoracic Duct","Lymph Nodes","Spleen","Thymus","Tonsils"]),
}
for sys_name, (defn, parts) in organ_systems.items():
    C(sys_name, defn, "DEFINITION", 0.98)
    R(sys_name, "Organ System", "IS_A", 0.99)
    for p in parts:
        if p not in [c["label"] for c in concepts]:
            C(p, f"anatomical structure, part of the {sys_name.lower()}", "FACT", 0.93)
        R(p, sys_name, "PART_OF", 0.95)

C("Organ System", "group of organs working together to perform a specific body function", "DEFINITION", 0.99)

# ============================================================
# CELL TYPES (100+)
# ============================================================
cell_types = {
    # Blood cells
    "Erythrocyte": ("red blood cell, biconcave disc carrying oxygen via hemoglobin, no nucleus in mature form", "Blood"),
    "Neutrophil": ("most abundant WBC, first responder to bacterial infection, forms NETs", "Blood"),
    "Eosinophil": ("WBC involved in parasitic defense and allergic reactions, contains eosinophilic granules", "Blood"),
    "Basophil": ("rarest WBC, releases histamine and heparin, involved in allergic responses", "Blood"),
    "Monocyte": ("large WBC that differentiates into macrophages and dendritic cells in tissues", "Blood"),
    "Lymphocyte": ("adaptive immune cell including T cells, B cells, and NK cells", "Blood"),
    "T Helper Cell": ("CD4+ lymphocyte coordinating immune response, target of HIV", "Immune System"),
    "Cytotoxic T Cell": ("CD8+ lymphocyte directly killing virus-infected and tumor cells via perforin/granzyme", "Immune System"),
    "Regulatory T Cell": ("CD4+CD25+FoxP3+ T cell suppressing immune response, maintaining self-tolerance", "Immune System"),
    "B Cell": ("lymphocyte producing antibodies, differentiates into plasma cells upon activation", "Immune System"),
    "Plasma Cell": ("terminally differentiated B cell secreting large quantities of antibodies", "Immune System"),
    "Memory B Cell": ("long-lived B cell providing rapid secondary immune response upon re-exposure", "Immune System"),
    "Natural Killer Cell": ("innate lymphoid cell killing virus-infected and tumor cells without prior sensitization", "Immune System"),
    "Dendritic Cell": ("professional antigen-presenting cell bridging innate and adaptive immunity", "Immune System"),
    "Macrophage": ("tissue-resident phagocyte derived from monocyte, antigen presentation and cytokine production", "Immune System"),
    "Mast Cell": ("tissue-resident cell releasing histamine and other mediators in allergic and inflammatory responses", "Immune System"),
    "Platelet": ("thrombocyte, anucleate cell fragment from megakaryocyte, essential for hemostasis", "Blood"),
    "Megakaryocyte": ("large bone marrow cell producing platelets through cytoplasmic fragmentation", "Bone Marrow"),
    "Hematopoietic Stem Cell": ("multipotent stem cell in bone marrow giving rise to all blood cell lineages", "Bone Marrow"),
    "Reticulocyte": ("immature red blood cell with residual RNA, indicates erythropoietic activity", "Bone Marrow"),
    # Epithelial
    "Squamous Epithelial Cell": ("flat thin cell lining surfaces for diffusion and filtration", "Epithelium"),
    "Cuboidal Epithelial Cell": ("cube-shaped cell in secretory and absorptive surfaces like kidney tubules", "Epithelium"),
    "Columnar Epithelial Cell": ("tall cell with oval nuclei, lines digestive tract for absorption and secretion", "Epithelium"),
    "Goblet Cell": ("mucus-secreting epithelial cell found in respiratory and GI tract", "Epithelium"),
    "Ciliated Epithelial Cell": ("cell with motile cilia moving mucus and particles, lines respiratory tract", "Epithelium"),
    "Pneumocyte Type I": ("thin alveolar cell covering 95% of alveolar surface, facilitates gas exchange", "Lungs"),
    "Pneumocyte Type II": ("alveolar cell secreting surfactant, progenitor for type I cells", "Lungs"),
    # Connective tissue
    "Fibroblast": ("main connective tissue cell producing collagen and extracellular matrix", "Connective Tissue"),
    "Chondrocyte": ("cartilage cell residing in lacunae, maintains cartilage matrix", "Cartilage"),
    "Osteoblast": ("bone-forming cell synthesizing osteoid matrix", "Bone"),
    "Osteoclast": ("large multinucleated cell resorbing bone, derived from monocyte lineage", "Bone"),
    "Osteocyte": ("mature bone cell embedded in lacunae, mechanosensing and mineral homeostasis", "Bone"),
    "Adipocyte": ("fat cell storing triglycerides, endocrine function secreting leptin and adiponectin", "Adipose Tissue"),
    # Muscle
    "Skeletal Muscle Fiber": ("multinucleated striated muscle cell under voluntary control", "Skeletal Muscle"),
    "Cardiac Myocyte": ("striated muscle cell of heart with intercalated discs for synchronized contraction", "Heart"),
    "Smooth Muscle Cell": ("spindle-shaped involuntary muscle cell in vessel walls and visceral organs", "Smooth Muscle"),
    # Nervous system
    "Neuron": ("electrically excitable cell transmitting information via action potentials and synaptic transmission", "Nervous System"),
    "Motor Neuron": ("neuron transmitting signals from CNS to muscles for voluntary movement", "Nervous System"),
    "Sensory Neuron": ("neuron detecting stimuli and transmitting sensory information to CNS", "Nervous System"),
    "Interneuron": ("neuron connecting sensory and motor neurons within CNS, processing information", "Nervous System"),
    "Purkinje Cell": ("large cerebellar neuron with extensive dendritic tree, inhibitory output of cerebellar cortex", "Cerebellum"),
    "Pyramidal Cell": ("excitatory neuron in cerebral cortex and hippocampus with pyramid-shaped soma", "Cerebrum"),
    "Astrocyte": ("star-shaped glial cell maintaining blood-brain barrier, neurotransmitter recycling", "Nervous System"),
    "Oligodendrocyte": ("CNS glial cell forming myelin sheaths around axons", "Nervous System"),
    "Schwann Cell": ("PNS glial cell forming myelin sheath around peripheral nerve axons", "Nervous System"),
    "Microglia": ("resident macrophage of CNS, immune surveillance and synaptic pruning", "Nervous System"),
    "Ependymal Cell": ("ciliated cell lining ventricles of brain, producing and circulating CSF", "Nervous System"),
    # GI specialized
    "Parietal Cell": ("gastric cell secreting HCl and intrinsic factor, target of proton pump inhibitors", "Stomach"),
    "Chief Cell": ("gastric cell secreting pepsinogen, the precursor of pepsin", "Stomach"),
    "G Cell": ("gastric antral cell secreting gastrin", "Stomach"),
    "Enterocyte": ("absorptive epithelial cell of intestinal villi with brush border microvilli", "Small Intestine"),
    "Paneth Cell": ("small intestinal crypt cell secreting antimicrobial defensins and lysozyme", "Small Intestine"),
    "Enteroendocrine Cell": ("GI epithelial cell secreting hormones like secretin, CCK, GLP-1", "Small Intestine"),
    "Hepatocyte": ("liver parenchymal cell performing metabolism, detoxification, bile production, protein synthesis", "Liver"),
    "Kupffer Cell": ("liver-resident macrophage in sinusoids, clearing pathogens and debris from portal blood", "Liver"),
    "Stellate Cell (Hepatic)": ("liver cell storing vitamin A, becomes myofibroblast in cirrhosis causing fibrosis", "Liver"),
    # Kidney
    "Podocyte": ("specialized cell wrapping around glomerular capillaries, forming filtration slits", "Glomerulus"),
    "Mesangial Cell": ("glomerular cell providing structural support and regulating blood flow", "Glomerulus"),
    "Juxtaglomerular Cell": ("modified smooth muscle cell in afferent arteriole secreting renin", "Kidney"),
    "Macula Densa Cell": ("specialized cell in distal tubule sensing NaCl concentration, regulating renin release", "Kidney"),
    # Endocrine
    "Pancreatic Beta Cell": ("islet cell producing insulin in response to elevated blood glucose", "Pancreatic Islets"),
    "Pancreatic Alpha Cell": ("islet cell producing glucagon in response to low blood glucose", "Pancreatic Islets"),
    "Pancreatic Delta Cell": ("islet cell producing somatostatin, paracrine inhibition of insulin and glucagon", "Pancreatic Islets"),
    "Thyroid Follicular Cell": ("cell synthesizing thyroglobulin and producing T3/T4 hormones", "Thyroid Gland"),
    "Parafollicular C Cell": ("thyroid cell producing calcitonin, origin of medullary thyroid carcinoma", "Thyroid Gland"),
    "Chromaffin Cell": ("adrenal medulla cell producing catecholamines (epinephrine, norepinephrine)", "Adrenal Medulla"),
    "Zona Glomerulosa Cell": ("adrenal cortex cell producing aldosterone", "Adrenal Cortex"),
    "Zona Fasciculata Cell": ("adrenal cortex cell producing cortisol", "Adrenal Cortex"),
    "Zona Reticularis Cell": ("adrenal cortex cell producing androgens (DHEA)", "Adrenal Cortex"),
    # Reproductive
    "Spermatogonium": ("stem cell in seminiferous tubules giving rise to spermatocytes via mitosis", "Testes"),
    "Spermatocyte": ("germ cell undergoing meiosis during spermatogenesis", "Testes"),
    "Spermatid": ("haploid cell from meiosis II that differentiates into spermatozoon", "Testes"),
    "Spermatozoon": ("mature male gamete with head (acrosome), midpiece (mitochondria), and flagellum", "Testes"),
    "Sertoli Cell": ("testicular nurse cell supporting spermatogenesis, forms blood-testis barrier, produces inhibin", "Testes"),
    "Leydig Cell": ("testicular interstitial cell producing testosterone in response to LH", "Testes"),
    "Oocyte": ("female germ cell, arrested in meiosis II until fertilization", "Ovary"),
    "Granulosa Cell": ("ovarian follicular cell producing estrogen, converts androgen via aromatase", "Ovary"),
    "Theca Cell": ("ovarian cell producing androgens in response to LH, substrate for granulosa aromatase", "Ovary"),
    # Skin
    "Keratinocyte": ("predominant epidermal cell producing keratin, forms skin barrier", "Epidermis"),
    "Melanocyte": ("neural crest-derived cell producing melanin pigment for UV protection", "Epidermis"),
    "Langerhans Cell": ("epidermal dendritic cell, antigen-presenting cell of the skin", "Epidermis"),
    "Merkel Cell": ("mechanoreceptor cell in basal epidermis for light touch sensation", "Epidermis"),
    # Other
    "Mesenchymal Stem Cell": ("multipotent stromal cell differentiating into bone, cartilage, fat, and muscle cells", "Connective Tissue"),
    "Satellite Cell": ("skeletal muscle stem cell for repair and regeneration after injury", "Skeletal Muscle"),
    "Pericyte": ("cell wrapping around capillaries, regulating blood flow and vessel stability", "Blood Vessels"),
    "Endothelial Cell": ("cell lining blood vessels, regulating vascular tone, permeability, and angiogenesis", "Blood Vessels"),
    "Red Pulp Macrophage": ("splenic macrophage recycling old erythrocytes and recovering iron", "Spleen"),
    "Follicular Dendritic Cell": ("non-hematopoietic cell in lymph node germinal centers presenting antigens to B cells", "Lymph Nodes"),
    "Myoepithelial Cell": ("contractile cell surrounding glandular acini aiding secretion", "Glands"),
    "Interstitial Cell of Cajal": ("pacemaker cell of GI tract generating slow waves for peristalsis", "Digestive System"),
    "Taste Receptor Cell": ("chemoreceptor cell in taste buds detecting sweet, salty, sour, bitter, umami", "Tongue"),
    "Photoreceptor Cell": ("retinal neuron converting light to electrical signals", "Retina"),
    "Rod Cell": ("photoreceptor for dim light and peripheral vision, contains rhodopsin", "Retina"),
    "Cone Cell": ("photoreceptor for color vision and visual acuity in bright light", "Retina"),
    "Hair Cell (Inner Ear)": ("mechanoreceptor in cochlea and vestibular system transducing sound/movement", "Inner Ear"),
    "Type I Pneumocyte": ("alternative name for pneumocyte type I, gas exchange surface", "Lungs"),
    "Clara Cell": ("club cell of bronchioles secreting surfactant components and detoxifying inhaled substances", "Bronchioles"),
}

for ct, (defn, location) in cell_types.items():
    C(ct, defn, "DEFINITION", 0.95)
    R(ct, "Cell Type", "IS_A", 0.98)
    R(ct, location, "PART_OF", 0.9)

C("Cell Type", "distinct type of cell with specific structure and function", "DEFINITION", 0.99)

# ============================================================
# ENZYMES (120+)
# ============================================================
enzymes = {
    # Digestive
    "Amylase": ("enzyme hydrolyzing starch to maltose, found in saliva and pancreas", "Starch Digestion"),
    "Pepsin": ("gastric protease active at pH 2, cleaves proteins into peptides", "Protein Digestion"),
    "Trypsin": ("pancreatic serine protease cleaving peptide bonds at lysine/arginine", "Protein Digestion"),
    "Chymotrypsin": ("pancreatic serine protease cleaving at hydrophobic amino acids", "Protein Digestion"),
    "Lipase": ("enzyme hydrolyzing triglycerides to fatty acids and glycerol", "Fat Digestion"),
    "Lactase": ("brush border enzyme hydrolyzing lactose to glucose and galactose", "Lactose Digestion"),
    "Sucrase": ("brush border enzyme hydrolyzing sucrose to glucose and fructose", "Sucrose Digestion"),
    "Maltase": ("brush border enzyme hydrolyzing maltose to two glucose molecules", "Maltose Digestion"),
    "Elastase": ("pancreatic protease cleaving elastin and other proteins", "Protein Digestion"),
    "Carboxypeptidase": ("pancreatic exopeptidase removing C-terminal amino acids", "Protein Digestion"),
    "Aminopeptidase": ("brush border exopeptidase removing N-terminal amino acids", "Protein Digestion"),
    "Enterokinase": ("duodenal enzyme converting trypsinogen to active trypsin", "Trypsin Activation"),
    "Phospholipase A2": ("enzyme cleaving phospholipids at sn-2 position, releases arachidonic acid", "Lipid Metabolism"),
    # Glycolysis
    "Hexokinase": ("enzyme phosphorylating glucose to glucose-6-phosphate, first step of glycolysis", "Glycolysis"),
    "Glucokinase": ("liver/beta-cell hexokinase isoform with high Km for glucose, glucose sensor", "Glycolysis"),
    "Phosphofructokinase-1": ("rate-limiting enzyme of glycolysis, allosterically regulated", "Glycolysis"),
    "Pyruvate Kinase": ("glycolytic enzyme converting PEP to pyruvate, generating ATP", "Glycolysis"),
    "Pyruvate Dehydrogenase": ("mitochondrial complex converting pyruvate to acetyl-CoA, requires thiamine", "Krebs Cycle"),
    "Aldolase": ("glycolytic enzyme splitting fructose-1,6-bisphosphate into G3P and DHAP", "Glycolysis"),
    # TCA / Oxidative
    "Citrate Synthase": ("first enzyme of TCA cycle, condenses acetyl-CoA with oxaloacetate", "Krebs Cycle"),
    "Isocitrate Dehydrogenase": ("TCA enzyme catalyzing oxidative decarboxylation of isocitrate, produces NADH", "Krebs Cycle"),
    "Alpha-Ketoglutarate Dehydrogenase": ("TCA complex similar to PDH, requires thiamine, rate-limiting", "Krebs Cycle"),
    "Succinate Dehydrogenase": ("TCA enzyme and Complex II of ETC, converts succinate to fumarate", "Krebs Cycle"),
    "ATP Synthase": ("Complex V of ETC, rotary enzyme synthesizing ATP from ADP using proton gradient", "Oxidative Phosphorylation"),
    "Cytochrome C Oxidase": ("Complex IV of ETC, final electron acceptor transferring electrons to O2", "Oxidative Phosphorylation"),
    # Gluconeogenesis
    "Glucose-6-Phosphatase": ("liver enzyme converting G6P to free glucose, final step of gluconeogenesis", "Gluconeogenesis"),
    "Fructose-1,6-Bisphosphatase": ("gluconeogenic enzyme bypassing PFK-1 reaction", "Gluconeogenesis"),
    "Pyruvate Carboxylase": ("mitochondrial enzyme converting pyruvate to oxaloacetate, requires biotin", "Gluconeogenesis"),
    "PEPCK": ("phosphoenolpyruvate carboxykinase, converts OAA to PEP in gluconeogenesis", "Gluconeogenesis"),
    # Lipid metabolism
    "Fatty Acid Synthase": ("multi-enzyme complex synthesizing palmitate from acetyl-CoA and malonyl-CoA", "Fatty Acid Synthesis"),
    "Acetyl-CoA Carboxylase": ("rate-limiting enzyme of fatty acid synthesis, converts acetyl-CoA to malonyl-CoA", "Fatty Acid Synthesis"),
    "HMG-CoA Reductase": ("rate-limiting enzyme of cholesterol synthesis, target of statins", "Cholesterol Synthesis"),
    "Lipoprotein Lipase": ("endothelial enzyme hydrolyzing triglycerides in lipoproteins, releases fatty acids", "Lipid Metabolism"),
    "Carnitine Palmitoyltransferase I": ("outer mitochondrial membrane enzyme, rate-limiting for beta-oxidation", "Beta-Oxidation"),
    "Hormone-Sensitive Lipase": ("adipocyte enzyme hydrolyzing stored triglycerides, activated by catecholamines", "Lipolysis"),
    # Nucleotide
    "Dihydrofolate Reductase": ("enzyme reducing dihydrofolate to tetrahydrofolate, target of methotrexate", "Nucleotide Synthesis"),
    "Thymidylate Synthase": ("enzyme synthesizing dTMP from dUMP, target of 5-fluorouracil", "DNA Synthesis"),
    "Ribonucleotide Reductase": ("enzyme converting ribonucleotides to deoxyribonucleotides for DNA synthesis", "DNA Synthesis"),
    "HGPRT": ("hypoxanthine-guanine phosphoribosyltransferase, purine salvage, deficient in Lesch-Nyhan", "Purine Salvage"),
    "Xanthine Oxidase": ("enzyme converting hypoxanthine to xanthine to uric acid, target of allopurinol", "Purine Catabolism"),
    # DNA/RNA
    "DNA Polymerase": ("enzyme synthesizing DNA from deoxyribonucleotides using template strand", "DNA Replication"),
    "RNA Polymerase II": ("enzyme transcribing mRNA from DNA template in eukaryotes", "Transcription"),
    "Reverse Transcriptase": ("viral enzyme synthesizing DNA from RNA template, used by retroviruses", "Viral Replication"),
    "Helicase": ("enzyme unwinding double-stranded DNA at replication fork", "DNA Replication"),
    "Topoisomerase": ("enzyme relieving torsional strain in DNA by cutting and rejoining strands", "DNA Replication"),
    "Telomerase": ("ribonucleoprotein extending telomeres, active in stem cells and cancer cells", "Telomere Maintenance"),
    "Ligase": ("enzyme joining DNA fragments by forming phosphodiester bonds", "DNA Repair"),
    # Signaling
    "Adenylyl Cyclase": ("enzyme converting ATP to cAMP, activated by Gs protein-coupled receptors", "Signal Transduction"),
    "Phospholipase C": ("enzyme cleaving PIP2 to IP3 and DAG, activated by Gq receptors", "Signal Transduction"),
    "Protein Kinase A": ("cAMP-dependent kinase phosphorylating serine/threonine residues", "Signal Transduction"),
    "Protein Kinase C": ("DAG-activated kinase in calcium signaling pathways", "Signal Transduction"),
    "Tyrosine Kinase": ("kinase phosphorylating tyrosine residues, important in growth factor signaling", "Signal Transduction"),
    "MAP Kinase": ("mitogen-activated protein kinase, key in cell proliferation signaling cascade", "Signal Transduction"),
    "PI3 Kinase": ("phosphoinositide 3-kinase, key enzyme in cell survival and growth signaling (AKT pathway)", "Signal Transduction"),
    "Phosphodiesterase": ("enzyme degrading cAMP and cGMP, target of sildenafil and milrinone", "Signal Transduction"),
    "Guanylyl Cyclase": ("enzyme producing cGMP from GTP, activated by nitric oxide and natriuretic peptides", "Signal Transduction"),
    # Coagulation
    "Thrombin": ("serine protease (Factor IIa) converting fibrinogen to fibrin, key in coagulation cascade", "Coagulation"),
    "Factor Xa": ("serine protease at convergence of intrinsic/extrinsic pathways, target of rivaroxaban", "Coagulation"),
    "Plasmin": ("serine protease dissolving fibrin clots in fibrinolysis", "Fibrinolysis"),
    "Tissue Plasminogen Activator": ("serine protease converting plasminogen to plasmin, used in stroke treatment", "Fibrinolysis"),
    # Inflammatory
    "Cyclooxygenase-1": ("constitutive COX producing prostaglandins for homeostatic functions", "Prostaglandin Synthesis"),
    "Cyclooxygenase-2": ("inducible COX producing prostaglandins in inflammation, target of NSAIDs", "Prostaglandin Synthesis"),
    "Lipoxygenase": ("enzyme producing leukotrienes from arachidonic acid, involved in asthma", "Leukotriene Synthesis"),
    "Phospholipase A2 (Inflammatory)": ("releases arachidonic acid from membrane phospholipids, inhibited by corticosteroids", "Inflammatory Cascade"),
    # Liver
    "Cytochrome P450": ("superfamily of liver enzymes metabolizing drugs, toxins, and endogenous compounds", "Drug Metabolism"),
    "CYP3A4": ("most abundant hepatic P450, metabolizes ~50% of drugs", "Drug Metabolism"),
    "CYP2D6": ("polymorphic P450 enzyme, metabolizes many psychiatric and cardiac drugs", "Drug Metabolism"),
    "UDP-Glucuronosyltransferase": ("phase II enzyme conjugating glucuronic acid to drugs for excretion", "Drug Metabolism"),
    "Alcohol Dehydrogenase": ("enzyme converting ethanol to acetaldehyde in liver", "Ethanol Metabolism"),
    "Aldehyde Dehydrogenase": ("enzyme converting acetaldehyde to acetate, inhibited by disulfiram", "Ethanol Metabolism"),
    "Glutathione S-Transferase": ("phase II enzyme conjugating glutathione to electrophilic compounds", "Detoxification"),
    # Other important
    "Angiotensin-Converting Enzyme": ("ACE, converts angiotensin I to angiotensin II, target of ACE inhibitors", "RAAS"),
    "Monoamine Oxidase": ("mitochondrial enzyme degrading monoamine neurotransmitters, target of MAOIs", "Neurotransmitter Metabolism"),
    "Catechol-O-Methyltransferase": ("enzyme degrading catecholamines, target of COMT inhibitors in Parkinson's", "Neurotransmitter Metabolism"),
    "Acetylcholinesterase": ("enzyme rapidly hydrolyzing acetylcholine at synapses, target of nerve agents", "Neurotransmitter Metabolism"),
    "Carbonic Anhydrase": ("enzyme catalyzing CO2 + H2O ↔ H2CO3, target of acetazolamide", "Acid-Base Balance"),
    "Aromatase": ("enzyme converting androgens to estrogens, target of aromatase inhibitors in breast cancer", "Steroidogenesis"),
    "5-Alpha Reductase": ("enzyme converting testosterone to DHT, target of finasteride", "Steroidogenesis"),
    "Superoxide Dismutase": ("antioxidant enzyme converting superoxide to hydrogen peroxide", "Antioxidant Defense"),
    "Catalase": ("enzyme decomposing hydrogen peroxide to water and oxygen", "Antioxidant Defense"),
    "Glutathione Peroxidase": ("selenium-containing enzyme reducing peroxides using glutathione", "Antioxidant Defense"),
    "Telomerase": ("ribonucleoprotein extending telomere repeats, active in cancer and stem cells", "Telomere Maintenance"),
    "Caspase-3": ("executioner caspase in apoptosis, cleaves cellular substrates", "Apoptosis"),
    "Caspase-9": ("initiator caspase in intrinsic apoptosis pathway, activated by cytochrome c", "Apoptosis"),
    "Caspase-8": ("initiator caspase in extrinsic apoptosis pathway, activated by death receptors", "Apoptosis"),
    "Matrix Metalloproteinase": ("zinc-dependent endopeptidase degrading ECM, involved in metastasis", "Tissue Remodeling"),
    "Lysozyme": ("enzyme hydrolyzing bacterial peptidoglycan cell wall, found in tears and saliva", "Innate Immunity"),
    "Myeloperoxidase": ("neutrophil enzyme producing hypochlorous acid for bactericidal activity", "Innate Immunity"),
    "NADPH Oxidase": ("enzyme complex generating superoxide in phagocytes, deficient in CGD", "Innate Immunity"),
    "Proteasome": ("large protein complex degrading ubiquitin-tagged proteins, target of bortezomib", "Protein Degradation"),
    "Ubiquitin Ligase": ("E3 enzyme tagging proteins with ubiquitin for proteasomal degradation", "Protein Degradation"),
    "Glycogen Synthase": ("enzyme polymerizing glucose into glycogen", "Glycogen Synthesis"),
    "Glycogen Phosphorylase": ("enzyme cleaving glucose-1-phosphate from glycogen", "Glycogenolysis"),
    "Glucose-6-Phosphate Dehydrogenase": ("rate-limiting enzyme of pentose phosphate pathway, deficiency causes hemolytic anemia", "Pentose Phosphate Pathway"),
    "Lactate Dehydrogenase": ("enzyme interconverting pyruvate and lactate, elevated in tissue damage", "Anaerobic Glycolysis"),
    "Creatine Kinase": ("enzyme catalyzing creatine ↔ phosphocreatine, elevated in MI and rhabdomyolysis", "Energy Metabolism"),
    "Alkaline Phosphatase": ("enzyme removing phosphate groups at alkaline pH, elevated in bone/liver disease", "Bone Metabolism"),
    "Acid Phosphatase": ("enzyme active at acid pH, prostatic acid phosphatase elevated in prostate cancer", "Prostate"),
    "Transaminase (ALT)": ("alanine aminotransferase, liver enzyme elevated in hepatocellular damage", "Liver Function"),
    "Transaminase (AST)": ("aspartate aminotransferase, found in liver and heart, elevated in damage", "Liver Function"),
    "Gamma-Glutamyl Transferase": ("liver enzyme induced by alcohol and drugs, elevated in cholestasis", "Liver Function"),
    "Troponin": ("cardiac regulatory protein complex, troponin I and T are specific biomarkers of MI", "Cardiac Biomarker"),
    "Nitric Oxide Synthase": ("enzyme producing NO from arginine; endothelial (eNOS), neuronal (nNOS), inducible (iNOS)", "Vasodilation"),
    "Prostaglandin Synthase": ("alternative name for COX enzymes producing prostaglandins from arachidonic acid", "Prostaglandin Synthesis"),
    "Renin Enzyme": ("aspartyl protease cleaving angiotensinogen, rate-limiting step in RAAS", "RAAS"),
    "Tryptophan Hydroxylase": ("rate-limiting enzyme in serotonin synthesis, converts tryptophan to 5-HTP", "Serotonin Synthesis"),
    "Tyrosine Hydroxylase": ("rate-limiting enzyme in catecholamine synthesis, converts tyrosine to L-DOPA", "Catecholamine Synthesis"),
    "DOPA Decarboxylase": ("enzyme converting L-DOPA to dopamine, also converts 5-HTP to serotonin", "Neurotransmitter Synthesis"),
    "Dopamine Beta-Hydroxylase": ("enzyme converting dopamine to norepinephrine in vesicles", "Catecholamine Synthesis"),
    "Phenylethanolamine N-Methyltransferase": ("adrenal enzyme converting norepinephrine to epinephrine", "Catecholamine Synthesis"),
    "Glutamic Acid Decarboxylase": ("enzyme converting glutamate to GABA, autoantigen in type 1 diabetes", "GABA Synthesis"),
}

for enz, (defn, pathway) in enzymes.items():
    C(enz, defn, "DEFINITION", 0.95)
    R(enz, "Enzyme", "IS_A", 0.98)
    R(enz, pathway, "USED_IN", 0.9)

C("Enzyme", "biological catalyst accelerating chemical reactions without being consumed", "DEFINITION", 0.99)

# Key enzyme inhibition relationships
enzyme_drug_targets = [
    ("HMG-CoA Reductase", "Statins", "INHIBITS"),
    ("Cyclooxygenase-2", "NSAIDs", "INHIBITS"),
    ("Cyclooxygenase-1", "Aspirin", "INHIBITS"),
    ("Angiotensin-Converting Enzyme", "ACE Inhibitors", "INHIBITS"),
    ("Xanthine Oxidase", "Allopurinol", "INHIBITS"),
    ("Dihydrofolate Reductase", "Methotrexate", "INHIBITS"),
    ("Thymidylate Synthase", "5-Fluorouracil", "INHIBITS"),
    ("Acetylcholinesterase", "Donepezil", "INHIBITS"),
    ("Monoamine Oxidase", "MAO Inhibitors", "INHIBITS"),
    ("Aromatase", "Aromatase Inhibitors", "INHIBITS"),
    ("5-Alpha Reductase", "Finasteride", "INHIBITS"),
    ("Phosphodiesterase", "Sildenafil", "INHIBITS"),
    ("Carbonic Anhydrase", "Acetazolamide", "INHIBITS"),
    ("Aldehyde Dehydrogenase", "Disulfiram", "INHIBITS"),
    ("Proteasome", "Bortezomib", "INHIBITS"),
    ("Reverse Transcriptase", "NRTIs", "INHIBITS"),
    ("Factor Xa", "Rivaroxaban", "INHIBITS"),
]
for enz, drug, rel in enzyme_drug_targets:
    if drug not in [c["label"] for c in concepts]:
        C(drug, f"drug/drug class that inhibits {enz}", "FACT", 0.94)
    R(drug, enz, "INHIBITS", 0.95)

# ============================================================
# DISEASES (500+)
# ============================================================
diseases = {
    # Cardiovascular (50)
    "Coronary Artery Disease": ("atherosclerotic narrowing of coronary arteries causing myocardial ischemia", "Cardiovascular System", ["Atherosclerosis"], ["Statins","Aspirin","Coronary Artery Bypass Graft","Percutaneous Coronary Intervention"]),
    "Myocardial Infarction": ("acute death of myocardium due to coronary artery occlusion", "Cardiovascular System", ["Coronary Artery Disease","Thrombus"], ["Percutaneous Coronary Intervention","Thrombolytics","Aspirin","Beta Blockers","ACE Inhibitors"]),
    "Heart Failure": ("inability of heart to pump sufficient blood for metabolic demands", "Cardiovascular System", ["Coronary Artery Disease","Hypertension","Cardiomyopathy"], ["ACE Inhibitors","Beta Blockers","Diuretics","SGLT2 Inhibitors"]),
    "Atrial Fibrillation": ("irregular rapid atrial activation causing ineffective atrial contraction, stroke risk", "Cardiovascular System", ["Hypertension","Heart Failure"], ["Anticoagulants","Rate Control","Rhythm Control","Catheter Ablation"]),
    "Hypertension": ("sustained elevated blood pressure >140/90 mmHg, major risk factor for CVD and stroke", "Cardiovascular System", [], ["ACE Inhibitors","ARBs","Calcium Channel Blockers","Thiazide Diuretics","Beta Blockers"]),
    "Aortic Stenosis": ("narrowing of aortic valve causing left ventricular pressure overload", "Cardiovascular System", ["Calcification","Bicuspid Aortic Valve"], ["Aortic Valve Replacement","TAVR"]),
    "Mitral Regurgitation": ("backflow of blood through mitral valve into left atrium during systole", "Cardiovascular System", ["Mitral Valve Prolapse","Rheumatic Heart Disease"], ["Mitral Valve Repair"]),
    "Aortic Dissection": ("tear in aortic intima allowing blood between vessel layers, surgical emergency", "Cardiovascular System", ["Hypertension","Marfan Syndrome"], ["Beta Blockers","Surgical Repair"]),
    "Pulmonary Embolism": ("obstruction of pulmonary arteries by thrombus, usually from DVT", "Cardiovascular System", ["Deep Vein Thrombosis"], ["Anticoagulants","Thrombolytics","Embolectomy"]),
    "Deep Vein Thrombosis": ("blood clot in deep veins, usually leg, risk of pulmonary embolism", "Cardiovascular System", ["Virchow Triad"], ["Anticoagulants"]),
    "Endocarditis": ("infection of heart valves, vegetations causing valve destruction and emboli", "Cardiovascular System", ["Staphylococcus aureus","Streptococcus viridans"], ["Antibiotics","Valve Replacement"]),
    "Pericarditis": ("inflammation of pericardium causing chest pain and friction rub", "Cardiovascular System", ["Viral Infection","Autoimmune"], ["NSAIDs","Colchicine"]),
    "Cardiac Tamponade": ("compression of heart by pericardial fluid, Beck triad: hypotension, JVD, muffled sounds", "Cardiovascular System", ["Pericardial Effusion"], ["Pericardiocentesis"]),
    "Dilated Cardiomyopathy": ("ventricular dilation with systolic dysfunction, most common cardiomyopathy", "Cardiovascular System", ["Alcohol","Viral Myocarditis","Genetic"], ["Heart Failure Treatment"]),
    "Hypertrophic Cardiomyopathy": ("asymmetric septal hypertrophy, most common cause of sudden cardiac death in young athletes", "Cardiovascular System", ["Genetic (Autosomal Dominant)"], ["Beta Blockers","Septal Myectomy","ICD"]),
    "Peripheral Artery Disease": ("atherosclerotic narrowing of peripheral arteries causing claudication", "Cardiovascular System", ["Atherosclerosis"], ["Exercise","Cilostazol","Revascularization"]),
    "Aortic Aneurysm": ("pathologic dilation of aorta >50% normal diameter, risk of rupture", "Cardiovascular System", ["Atherosclerosis","Hypertension"], ["Surgical Repair","Endovascular Repair"]),
    "Rheumatic Heart Disease": ("valvular damage from rheumatic fever following Group A Strep pharyngitis", "Cardiovascular System", ["Rheumatic Fever"], ["Penicillin Prophylaxis","Valve Replacement"]),
    "Wolff-Parkinson-White Syndrome": ("accessory pathway (Bundle of Kent) causing pre-excitation and tachyarrhythmias", "Cardiovascular System", ["Congenital Accessory Pathway"], ["Catheter Ablation"]),
    
    # Respiratory (40)
    "Asthma": ("chronic airway inflammation with reversible bronchoconstriction, wheezing, and dyspnea", "Respiratory System", ["Allergens","Genetic Predisposition"], ["Inhaled Corticosteroids","Short-Acting Beta Agonists","Long-Acting Beta Agonists","Leukotriene Antagonists"]),
    "COPD": ("chronic obstructive pulmonary disease, irreversible airflow limitation from emphysema/chronic bronchitis", "Respiratory System", ["Smoking"], ["Bronchodilators","Inhaled Corticosteroids","Oxygen Therapy","Smoking Cessation"]),
    "Emphysema": ("destruction of alveolar walls causing air trapping and decreased gas exchange surface", "Respiratory System", ["Smoking","Alpha-1 Antitrypsin Deficiency"], ["Smoking Cessation","Bronchodilators"]),
    "Chronic Bronchitis": ("productive cough for 3+ months in 2+ consecutive years, mucus hypersecretion", "Respiratory System", ["Smoking"], ["Smoking Cessation","Bronchodilators"]),
    "Pneumonia": ("infection of lung parenchyma causing consolidation and impaired gas exchange", "Respiratory System", ["Streptococcus pneumoniae","Viral","Aspiration"], ["Antibiotics","Antivirals"]),
    "Tuberculosis": ("Mycobacterium tuberculosis infection forming caseating granulomas, primarily in lungs", "Respiratory System", ["Mycobacterium tuberculosis"], ["RIPE Therapy"]),
    "Lung Cancer": ("malignant neoplasm of lung, leading cause of cancer death; small cell and non-small cell types", "Respiratory System", ["Smoking","Radon","Asbestos"], ["Surgery","Chemotherapy","Radiation","Immunotherapy","Targeted Therapy"]),
    "Pulmonary Fibrosis": ("progressive scarring of lung interstitium reducing gas exchange and lung compliance", "Respiratory System", ["Idiopathic","Occupational Exposure","Autoimmune"], ["Pirfenidone","Nintedanib","Lung Transplant"]),
    "Pneumothorax": ("air in pleural space causing lung collapse", "Respiratory System", ["Trauma","Spontaneous"], ["Chest Tube","Observation"]),
    "Pleural Effusion": ("abnormal fluid accumulation in pleural space", "Respiratory System", ["Heart Failure","Pneumonia","Malignancy"], ["Thoracentesis","Chest Tube"]),
    "Acute Respiratory Distress Syndrome": ("severe inflammatory lung injury with bilateral infiltrates and refractory hypoxemia", "Respiratory System", ["Sepsis","Pneumonia","Trauma"], ["Mechanical Ventilation","Prone Positioning"]),
    "Cystic Fibrosis": ("autosomal recessive CFTR mutation causing thick secretions in lungs, pancreas, and GI tract", "Respiratory System", ["CFTR Gene Mutation"], ["CFTR Modulators","Chest Physiotherapy","Pancreatic Enzymes"]),
    "Obstructive Sleep Apnea": ("repeated upper airway collapse during sleep causing hypoxia and sleep fragmentation", "Respiratory System", ["Obesity","Anatomical Narrowing"], ["CPAP","Weight Loss"]),
    "Sarcoidosis": ("systemic granulomatous disease of unknown cause, bilateral hilar lymphadenopathy, non-caseating granulomas", "Respiratory System", ["Unknown"], ["Corticosteroids"]),
    "Mesothelioma": ("malignancy of pleura strongly associated with asbestos exposure", "Respiratory System", ["Asbestos Exposure"], ["Chemotherapy","Surgery"]),
    "Pulmonary Hypertension": ("elevated pulmonary artery pressure >20 mmHg causing right heart failure", "Cardiovascular System", ["Idiopathic","Left Heart Disease","Lung Disease"], ["Prostacyclin Analogs","PDE5 Inhibitors","Endothelin Receptor Antagonists"]),
    
    # Neurological (60)
    "Stroke (Ischemic)": ("acute brain infarction from arterial occlusion, most common stroke type", "Nervous System", ["Atherosclerosis","Atrial Fibrillation","Thrombus"], ["tPA","Thrombectomy","Aspirin"]),
    "Stroke (Hemorrhagic)": ("brain hemorrhage from ruptured vessel, intracerebral or subarachnoid", "Nervous System", ["Hypertension","Aneurysm Rupture"], ["Blood Pressure Control","Surgical Evacuation"]),
    "Alzheimer Disease": ("progressive neurodegenerative dementia with amyloid plaques and neurofibrillary tangles", "Nervous System", ["Amyloid Beta Accumulation","Tau Protein"], ["Cholinesterase Inhibitors","Memantine","Lecanemab"]),
    "Parkinson Disease": ("neurodegenerative disorder with loss of dopaminergic neurons in substantia nigra", "Nervous System", ["Dopaminergic Neuron Loss"], ["Levodopa/Carbidopa","Dopamine Agonists","MAO-B Inhibitors","Deep Brain Stimulation"]),
    "Multiple Sclerosis": ("autoimmune demyelination of CNS white matter, relapsing-remitting or progressive", "Nervous System", ["Autoimmune Demyelination"], ["Interferon Beta","Natalizumab","Ocrelizumab","Fingolimod"]),
    "Epilepsy": ("recurrent unprovoked seizures due to abnormal neuronal electrical activity", "Nervous System", ["Genetic","Structural Brain Lesion"], ["Anticonvulsants","Surgery"]),
    "Migraine": ("recurrent headache with aura, photophobia, nausea; neurovascular pathophysiology", "Nervous System", ["Cortical Spreading Depression","CGRP Release"], ["Triptans","CGRP Antagonists","Beta Blockers","Topiramate"]),
    "Meningitis": ("inflammation of meninges, bacterial form is medical emergency", "Nervous System", ["Neisseria meningitidis","Streptococcus pneumoniae","Viral"], ["Antibiotics","Dexamethasone","Antivirals"]),
    "Encephalitis": ("brain parenchyma inflammation, often viral (HSV most common sporadic cause)", "Nervous System", ["Herpes Simplex Virus","Autoimmune"], ["Acyclovir","Immunotherapy"]),
    "Guillain-Barre Syndrome": ("acute inflammatory demyelinating polyneuropathy, ascending paralysis post-infection", "Nervous System", ["Post-Infectious Autoimmune"], ["IVIG","Plasmapheresis"]),
    "Amyotrophic Lateral Sclerosis": ("progressive degeneration of upper and lower motor neurons, fatal", "Nervous System", ["Unknown","SOD1 Mutation"], ["Riluzole","Edaravone"]),
    "Huntington Disease": ("autosomal dominant trinucleotide repeat (CAG) in huntingtin gene, chorea and dementia", "Nervous System", ["HTT Gene Mutation"], ["Tetrabenazine","Supportive Care"]),
    "Myasthenia Gravis": ("autoimmune antibodies against acetylcholine receptors at NMJ, fatigable weakness", "Nervous System", ["Anti-AChR Antibodies"], ["Cholinesterase Inhibitors","Immunosuppression","Thymectomy"]),
    "Bell Palsy": ("acute unilateral facial nerve (CN VII) paralysis, often idiopathic/viral", "Nervous System", ["HSV Reactivation"], ["Corticosteroids","Eye Protection"]),
    "Trigeminal Neuralgia": ("severe episodic facial pain in trigeminal nerve distribution", "Nervous System", ["Vascular Compression"], ["Carbamazepine","Microvascular Decompression"]),
    "Hydrocephalus": ("excessive CSF accumulation in ventricles causing increased intracranial pressure", "Nervous System", ["Obstruction","Impaired Absorption"], ["Ventriculoperitoneal Shunt","Endoscopic Third Ventriculostomy"]),
    "Glioblastoma": ("grade IV astrocytoma, most aggressive primary brain tumor, poor prognosis", "Nervous System", ["Unknown"], ["Surgery","Temozolomide","Radiation"]),
    "Essential Tremor": ("most common movement disorder, postural/action tremor, often familial", "Nervous System", ["Genetic"], ["Propranolol","Primidone"]),
    "Narcolepsy": ("sleep disorder with excessive daytime sleepiness and cataplexy, orexin deficiency", "Nervous System", ["Orexin Neuron Loss"], ["Modafinil","Sodium Oxybate"]),
    "Spinal Cord Injury": ("damage to spinal cord causing paralysis and sensory loss below injury level", "Nervous System", ["Trauma"], ["Surgical Stabilization","Rehabilitation"]),
    "Subarachnoid Hemorrhage": ("bleeding into subarachnoid space, usually from ruptured berry aneurysm, thunderclap headache", "Nervous System", ["Berry Aneurysm Rupture"], ["Surgical Clipping","Endovascular Coiling","Nimodipine"]),
    "Cerebral Palsy": ("non-progressive motor disorder from perinatal brain injury", "Nervous System", ["Perinatal Hypoxia","Prematurity"], ["Physical Therapy","Botulinum Toxin","Surgery"]),
    "Neuropathy (Peripheral)": ("damage to peripheral nerves causing numbness, tingling, weakness", "Nervous System", ["Diabetes","Alcohol","B12 Deficiency"], ["Treat Underlying Cause","Gabapentin","Duloxetine"]),
    "Carpal Tunnel Syndrome": ("median nerve compression at wrist causing hand numbness and weakness", "Nervous System", ["Repetitive Use","Pregnancy","Hypothyroidism"], ["Splinting","Corticosteroid Injection","Carpal Tunnel Release"]),
    
    # Endocrine (40)
    "Type 1 Diabetes": ("autoimmune destruction of pancreatic beta cells causing absolute insulin deficiency", "Endocrine System", ["Autoimmune Beta Cell Destruction"], ["Insulin Therapy"]),
    "Type 2 Diabetes": ("insulin resistance with relative insulin deficiency, most common form of diabetes", "Endocrine System", ["Insulin Resistance","Obesity"], ["Metformin","SGLT2 Inhibitors","GLP-1 Agonists","Insulin Therapy"]),
    "Diabetic Ketoacidosis": ("metabolic emergency in diabetes with hyperglycemia, ketosis, and acidosis", "Endocrine System", ["Insulin Deficiency"], ["Insulin","IV Fluids","Electrolyte Replacement"]),
    "Hyperthyroidism": ("excess thyroid hormone causing weight loss, tachycardia, heat intolerance, tremor", "Endocrine System", ["Graves Disease","Toxic Nodule"], ["Methimazole","Radioactive Iodine","Thyroidectomy","Beta Blockers"]),
    "Hypothyroidism": ("insufficient thyroid hormone causing fatigue, weight gain, cold intolerance, constipation", "Endocrine System", ["Hashimoto Thyroiditis","Iodine Deficiency"], ["Levothyroxine"]),
    "Graves Disease": ("autoimmune hyperthyroidism with TSH receptor-stimulating antibodies, exophthalmos", "Endocrine System", ["TSI Antibodies"], ["Methimazole","Radioactive Iodine","Thyroidectomy"]),
    "Hashimoto Thyroiditis": ("autoimmune chronic lymphocytic thyroiditis, most common cause of hypothyroidism", "Endocrine System", ["Anti-TPO Antibodies"], ["Levothyroxine"]),
    "Cushing Syndrome": ("excess cortisol causing moon face, buffalo hump, striae, hypertension, hyperglycemia", "Endocrine System", ["Pituitary Adenoma","Exogenous Steroids","Adrenal Tumor"], ["Surgery","Ketoconazole"]),
    "Addison Disease": ("primary adrenal insufficiency with cortisol and aldosterone deficiency, hyperpigmentation", "Endocrine System", ["Autoimmune Adrenal Destruction"], ["Hydrocortisone","Fludrocortisone"]),
    "Pheochromocytoma": ("catecholamine-secreting tumor of adrenal medulla causing episodic hypertension", "Endocrine System", ["Chromaffin Cell Tumor"], ["Alpha Blockers then Beta Blockers","Adrenalectomy"]),
    "Hyperaldosteronism": ("excess aldosterone causing hypertension and hypokalemia, Conn syndrome", "Endocrine System", ["Adrenal Adenoma","Bilateral Hyperplasia"], ["Spironolactone","Adrenalectomy"]),
    "Acromegaly": ("excess growth hormone in adults causing enlarged hands/feet/jaw, often pituitary adenoma", "Endocrine System", ["Pituitary Adenoma"], ["Transsphenoidal Surgery","Octreotide","Pegvisomant"]),
    "Diabetes Insipidus": ("excessive dilute urine from ADH deficiency (central) or resistance (nephrogenic)", "Endocrine System", ["ADH Deficiency","ADH Resistance"], ["Desmopressin","Thiazide Diuretics"]),
    "SIADH": ("syndrome of inappropriate ADH secretion causing dilutional hyponatremia", "Endocrine System", ["Lung Cancer","CNS Disease","Drugs"], ["Fluid Restriction","Tolvaptan"]),
    "Prolactinoma": ("most common pituitary adenoma, secretes prolactin causing galactorrhea and amenorrhea", "Endocrine System", ["Pituitary Adenoma"], ["Cabergoline","Bromocriptine"]),
    "Hyperparathyroidism": ("excess PTH causing hypercalcemia, bones/stones/groans/psychiatric moans", "Endocrine System", ["Parathyroid Adenoma"], ["Parathyroidectomy"]),
    "Hypoparathyroidism": ("PTH deficiency causing hypocalcemia, tetany, Chvostek/Trousseau signs", "Endocrine System", ["Surgical Removal","Autoimmune"], ["Calcium","Vitamin D Supplements"]),
    "Thyroid Cancer": ("malignant neoplasm of thyroid; papillary most common, good prognosis", "Endocrine System", ["Radiation Exposure","Genetic"], ["Thyroidectomy","Radioactive Iodine","TSH Suppression"]),
    "Multiple Endocrine Neoplasia": ("inherited syndromes (MEN1, MEN2) with tumors in multiple endocrine glands", "Endocrine System", ["Genetic (MEN1/RET Mutations)"], ["Surveillance","Prophylactic Thyroidectomy"]),
    "Metabolic Syndrome": ("cluster of obesity, hypertension, hyperglycemia, dyslipidemia increasing CVD risk", "Endocrine System", ["Insulin Resistance","Obesity"], ["Lifestyle Modification","Metformin"]),
    "Polycystic Ovary Syndrome": ("hormonal disorder with hyperandrogenism, oligo-ovulation, polycystic ovaries", "Endocrine System", ["Insulin Resistance","Hyperandrogenism"], ["Combined OCP","Metformin","Spironolactone"]),
    
    # GI (50)
    "Gastroesophageal Reflux Disease": ("chronic acid reflux causing heartburn and esophageal mucosal damage", "Digestive System", ["Lower Esophageal Sphincter Dysfunction"], ["Proton Pump Inhibitors","H2 Blockers","Fundoplication"]),
    "Peptic Ulcer Disease": ("mucosal erosions in stomach or duodenum, H. pylori or NSAIDs", "Digestive System", ["Helicobacter pylori","NSAIDs"], ["Proton Pump Inhibitors","H. pylori Eradication","Avoid NSAIDs"]),
    "Celiac Disease": ("autoimmune enteropathy triggered by gluten causing villous atrophy in small intestine", "Digestive System", ["Gluten","HLA-DQ2/DQ8"], ["Gluten-Free Diet"]),
    "Crohn Disease": ("transmural granulomatous inflammation anywhere in GI tract, skip lesions, fistulas", "Digestive System", ["Autoimmune","Genetic"], ["Corticosteroids","Biologics","Immunomodulators","Surgery"]),
    "Ulcerative Colitis": ("chronic mucosal inflammation of colon and rectum, continuous from rectum, bloody diarrhea", "Digestive System", ["Autoimmune"], ["5-ASA","Corticosteroids","Biologics","Colectomy"]),
    "Irritable Bowel Syndrome": ("functional GI disorder with abdominal pain and altered bowel habits, no structural cause", "Digestive System", ["Gut-Brain Axis Dysregulation"], ["Dietary Modification","Antispasmodics","SSRIs"]),
    "Appendicitis": ("acute inflammation of appendix, RLQ pain, surgical emergency", "Digestive System", ["Appendiceal Obstruction"], ["Appendectomy"]),
    "Diverticulitis": ("inflammation/infection of colonic diverticula, LLQ pain", "Digestive System", ["Diverticulosis","Low Fiber Diet"], ["Antibiotics","Surgery"]),
    "Cholecystitis": ("gallbladder inflammation usually from gallstone obstruction of cystic duct", "Digestive System", ["Gallstones"], ["Cholecystectomy","Antibiotics"]),
    "Cholelithiasis": ("gallstones in gallbladder, cholesterol or pigment stones", "Digestive System", ["4F Risk Factors"], ["Cholecystectomy"]),
    "Pancreatitis (Acute)": ("acute pancreatic inflammation, usually from gallstones or alcohol, epigastric pain", "Digestive System", ["Gallstones","Alcohol"], ["NPO","IV Fluids","Pain Management"]),
    "Pancreatitis (Chronic)": ("progressive pancreatic fibrosis and exocrine/endocrine insufficiency", "Digestive System", ["Chronic Alcohol Use"], ["Pancreatic Enzyme Replacement","Pain Management"]),
    "Hepatitis A": ("acute self-limited viral hepatitis transmitted fecal-oral, HAV", "Digestive System", ["Hepatitis A Virus"], ["Supportive Care","Vaccination"]),
    "Hepatitis B": ("viral hepatitis transmitted via blood/body fluids, can become chronic, HBV", "Digestive System", ["Hepatitis B Virus"], ["Tenofovir","Entecavir","Interferon","Vaccination"]),
    "Hepatitis C": ("viral hepatitis transmitted via blood, chronic in 75%, leading cause of liver transplant", "Digestive System", ["Hepatitis C Virus"], ["Direct-Acting Antivirals"]),
    "Cirrhosis": ("end-stage liver fibrosis with nodular regeneration, portal hypertension, liver failure", "Digestive System", ["Chronic Hepatitis","Alcohol","NAFLD"], ["Treat Underlying Cause","Liver Transplant"]),
    "Hepatocellular Carcinoma": ("primary liver cancer arising in setting of cirrhosis, AFP elevation", "Digestive System", ["Cirrhosis","Hepatitis B","Hepatitis C"], ["Surgery","Ablation","Sorafenib","Liver Transplant"]),
    "Colorectal Cancer": ("malignancy of colon or rectum, adenoma-carcinoma sequence", "Digestive System", ["APC/KRAS Mutations","Lynch Syndrome","Polyps"], ["Surgery","Chemotherapy","Screening Colonoscopy"]),
    "Gastric Cancer": ("malignancy of stomach, intestinal and diffuse types, H. pylori associated", "Digestive System", ["Helicobacter pylori","Dietary Factors"], ["Gastrectomy","Chemotherapy"]),
    "Esophageal Cancer": ("squamous cell (alcohol/smoking) or adenocarcinoma (Barrett esophagus)", "Digestive System", ["Barrett Esophagus","Smoking","Alcohol"], ["Esophagectomy","Chemoradiation"]),
    "Pancreatic Cancer": ("adenocarcinoma of pancreas, poor prognosis, painless jaundice if head involvement", "Digestive System", ["Smoking","Chronic Pancreatitis","Genetic"], ["Whipple Procedure","Chemotherapy"]),
    "Liver Failure (Acute)": ("rapid hepatic dysfunction with coagulopathy and encephalopathy", "Digestive System", ["Acetaminophen Overdose","Viral Hepatitis"], ["N-Acetylcysteine","Liver Transplant"]),
    "Portal Hypertension": ("elevated portal venous pressure causing varices, ascites, splenomegaly", "Digestive System", ["Cirrhosis"], ["Beta Blockers","Band Ligation","TIPS"]),
    "Intestinal Obstruction": ("mechanical or functional blockage of intestinal contents", "Digestive System", ["Adhesions","Hernia","Tumor"], ["NGT Decompression","Surgery"]),
    "Achalasia": ("esophageal motility disorder with LES failure to relax and absent peristalsis", "Digestive System", ["Loss of Myenteric Neurons"], ["Pneumatic Dilation","Heller Myotomy"]),
    "Barrett Esophagus": ("intestinal metaplasia of esophageal squamous epithelium from chronic GERD, premalignant", "Digestive System", ["Chronic GERD"], ["Proton Pump Inhibitors","Surveillance Endoscopy"]),
    "NAFLD": ("non-alcoholic fatty liver disease, hepatic steatosis without significant alcohol use", "Digestive System", ["Obesity","Insulin Resistance"], ["Weight Loss","Exercise"]),
    "Wilson Disease": ("autosomal recessive copper accumulation in liver and brain, ATP7B mutation", "Digestive System", ["ATP7B Gene Mutation"], ["Penicillamine","Zinc","Liver Transplant"]),
    "Hemochromatosis": ("iron overload disorder, usually HFE mutation, damages liver/heart/pancreas", "Digestive System", ["HFE Gene Mutation"], ["Phlebotomy","Iron Chelation"]),
    
    # Hematologic/Oncologic (50)
    "Iron Deficiency Anemia": ("microcytic hypochromic anemia from iron depletion, most common anemia worldwide", "Immune System", ["Blood Loss","Poor Intake","Malabsorption"], ["Iron Supplementation","Treat Underlying Cause"]),
    "Vitamin B12 Deficiency Anemia": ("megaloblastic anemia with hypersegmented neutrophils, neurologic symptoms", "Immune System", ["Pernicious Anemia","Malabsorption"], ["B12 Supplementation"]),
    "Folate Deficiency Anemia": ("megaloblastic anemia without neurologic symptoms, common in pregnancy and alcoholism", "Immune System", ["Poor Intake","Alcoholism"], ["Folate Supplementation"]),
    "Sickle Cell Disease": ("autosomal recessive hemoglobinopathy (HbS), vaso-occlusive crises, chronic hemolysis", "Immune System", ["HBB Gene Mutation"], ["Hydroxyurea","Pain Management","Blood Transfusion","Stem Cell Transplant"]),
    "Thalassemia": ("inherited hemoglobin disorder with decreased alpha or beta globin chain production", "Immune System", ["Alpha/Beta Globin Gene Mutations"], ["Blood Transfusion","Iron Chelation","Stem Cell Transplant"]),
    "Hemophilia A": ("X-linked recessive Factor VIII deficiency causing bleeding diathesis", "Immune System", ["Factor VIII Deficiency"], ["Factor VIII Replacement","Emicizumab"]),
    "Hemophilia B": ("X-linked recessive Factor IX deficiency (Christmas disease)", "Immune System", ["Factor IX Deficiency"], ["Factor IX Replacement"]),
    "Von Willebrand Disease": ("most common inherited bleeding disorder, deficient/dysfunctional vWF", "Immune System", ["VWF Gene Mutation"], ["Desmopressin","VWF Replacement"]),
    "Disseminated Intravascular Coagulation": ("pathologic activation of coagulation causing thrombosis and bleeding simultaneously", "Immune System", ["Sepsis","Malignancy","Obstetric Complications"], ["Treat Underlying Cause","Blood Products"]),
    "Thrombotic Thrombocytopenic Purpura": ("ADAMTS13 deficiency causing microangiopathic hemolytic anemia and thrombocytopenia", "Immune System", ["ADAMTS13 Deficiency"], ["Plasmapheresis","Caplacizumab"]),
    "Immune Thrombocytopenia": ("autoimmune platelet destruction causing petechiae and bleeding", "Immune System", ["Antiplatelet Antibodies"], ["Corticosteroids","IVIG","Splenectomy","Romiplostim"]),
    "Acute Lymphoblastic Leukemia": ("most common childhood cancer, malignant lymphoid precursor proliferation", "Immune System", ["Genetic Mutations"], ["Chemotherapy","Stem Cell Transplant","CAR-T Therapy"]),
    "Acute Myeloid Leukemia": ("malignant myeloid precursor proliferation, Auer rods, adults", "Immune System", ["Genetic Mutations","Myelodysplasia"], ["Chemotherapy","Stem Cell Transplant"]),
    "Chronic Lymphocytic Leukemia": ("most common adult leukemia, mature B-cell proliferation, smudge cells", "Immune System", ["Unknown"], ["Ibrutinib","Venetoclax","Rituximab"]),
    "Chronic Myeloid Leukemia": ("BCR-ABL fusion (Philadelphia chromosome) causing myeloid proliferation", "Immune System", ["BCR-ABL Fusion"], ["Imatinib","Dasatinib"]),
    "Hodgkin Lymphoma": ("lymphoma with Reed-Sternberg cells, bimodal age distribution, good prognosis", "Immune System", ["EBV Associated"], ["ABVD Chemotherapy","Radiation"]),
    "Non-Hodgkin Lymphoma": ("heterogeneous group of lymphoid malignancies, B-cell or T-cell origin", "Immune System", ["Various"], ["R-CHOP","Radiation","Targeted Therapy"]),
    "Multiple Myeloma": ("malignant plasma cell neoplasm producing monoclonal immunoglobulin, CRAB criteria", "Immune System", ["Plasma Cell Proliferation"], ["Bortezomib","Lenalidomide","Dexamethasone","Stem Cell Transplant"]),
    "Polycythemia Vera": ("JAK2 mutation causing erythrocytosis, risk of thrombosis", "Immune System", ["JAK2 V617F Mutation"], ["Phlebotomy","Hydroxyurea","Aspirin"]),
    "Myelodysplastic Syndrome": ("clonal hematopoietic disorder with ineffective hematopoiesis and dysplasia", "Immune System", ["Genetic Mutations"], ["Supportive Care","Azacitidine","Stem Cell Transplant"]),
    "Aplastic Anemia": ("pancytopenia from bone marrow failure, hypocellular marrow", "Immune System", ["Autoimmune","Drugs","Radiation"], ["Immunosuppression","Stem Cell Transplant"]),
    "Glucose-6-Phosphate Dehydrogenase Deficiency": ("X-linked enzyme deficiency causing episodic hemolytic anemia with oxidative stress", "Immune System", ["G6PD Gene Mutation"], ["Avoid Triggers"]),
    "Hereditary Spherocytosis": ("spectrin/ankyrin defect causing spherical RBCs with increased osmotic fragility", "Immune System", ["Spectrin/Ankyrin Mutations"], ["Splenectomy","Folate"]),
    "Breast Cancer": ("most common cancer in women, ductal or lobular, hormone receptor status guides treatment", "Reproductive System", ["BRCA1/BRCA2","Estrogen Exposure"], ["Surgery","Chemotherapy","Radiation","Tamoxifen","Trastuzumab"]),
    "Prostate Cancer": ("most common cancer in men, adenocarcinoma, PSA screening", "Reproductive System", ["Age","Family History"], ["Active Surveillance","Prostatectomy","Radiation","ADT"]),
    "Cervical Cancer": ("HPV-associated malignancy of cervix, prevented by vaccination and screening", "Reproductive System", ["HPV Infection"], ["Surgery","Chemoradiation","HPV Vaccination"]),
    "Ovarian Cancer": ("malignancy of ovary, often diagnosed late, CA-125 marker", "Reproductive System", ["BRCA Mutations","Age"], ["Surgery","Platinum-Based Chemotherapy","PARP Inhibitors"]),
    "Testicular Cancer": ("germ cell tumor in young men, seminoma or non-seminoma, highly curable", "Reproductive System", ["Cryptorchidism","Genetic"], ["Orchiectomy","Chemotherapy","Radiation"]),
    "Melanoma": ("malignant neoplasm of melanocytes, ABCDE criteria, rapidly metastatic", "Integumentary System", ["UV Exposure","Genetic"], ["Surgical Excision","Immunotherapy","Targeted Therapy"]),
    "Leukemia (General)": ("cancer of blood-forming tissues with abnormal white blood cell proliferation", "Immune System", ["Genetic Mutations","Radiation"], ["Chemotherapy","Stem Cell Transplant"]),
    
    # Renal (30)
    "Acute Kidney Injury": ("sudden decline in renal function with rising creatinine and oliguria", "Urinary System", ["Prerenal","Intrinsic","Postrenal"], ["Treat Underlying Cause","Dialysis"]),
    "Chronic Kidney Disease": ("progressive irreversible loss of renal function over months to years", "Urinary System", ["Diabetes","Hypertension","Glomerulonephritis"], ["ACE Inhibitors","Blood Pressure Control","Dialysis","Kidney Transplant"]),
    "Nephrotic Syndrome": ("proteinuria >3.5g/day, hypoalbuminemia, edema, hyperlipidemia", "Urinary System", ["Minimal Change Disease","Membranous Nephropathy","FSGS"], ["Corticosteroids","Immunosuppression"]),
    "Nephritic Syndrome": ("hematuria, RBC casts, hypertension, mild proteinuria from glomerular inflammation", "Urinary System", ["Post-Streptococcal GN","IgA Nephropathy","ANCA Vasculitis"], ["Immunosuppression","Supportive Care"]),
    "Kidney Stones": ("nephrolithiasis, calcium oxalate most common, colicky flank pain", "Urinary System", ["Hypercalciuria","Dehydration","Hyperoxaluria"], ["Hydration","Pain Management","Lithotripsy","Ureteroscopy"]),
    "Urinary Tract Infection": ("bacterial infection of urinary tract, E. coli most common cause", "Urinary System", ["E. coli","Urinary Stasis"], ["Antibiotics"]),
    "Pyelonephritis": ("bacterial infection of renal parenchyma, flank pain and fever", "Urinary System", ["Ascending UTI"], ["IV Antibiotics"]),
    "Polycystic Kidney Disease": ("inherited disorder with bilateral renal cysts, ADPKD most common", "Urinary System", ["PKD1/PKD2 Mutations"], ["Tolvaptan","Blood Pressure Control","Dialysis"]),
    "Renal Cell Carcinoma": ("most common renal malignancy in adults, classic triad: hematuria, flank pain, mass", "Urinary System", ["VHL Mutation","Smoking"], ["Nephrectomy","Targeted Therapy","Immunotherapy"]),
    "Glomerulonephritis": ("inflammation of glomeruli causing hematuria and proteinuria", "Urinary System", ["Immune Complex Deposition","Anti-GBM"], ["Immunosuppression"]),
    "Diabetic Nephropathy": ("progressive kidney damage from diabetes, leading cause of ESRD", "Urinary System", ["Diabetes","Hyperglycemia"], ["ACE Inhibitors","Glycemic Control","SGLT2 Inhibitors"]),
    "Renal Tubular Acidosis": ("impaired renal acid excretion causing non-anion gap metabolic acidosis", "Urinary System", ["Type 1/2/4 Defects"], ["Bicarbonate","Treat Underlying Cause"]),
    "Hydronephrosis": ("dilation of renal pelvis from urinary obstruction", "Urinary System", ["Kidney Stones","BPH","Tumor"], ["Relieve Obstruction"]),
    
    # Musculoskeletal (30)
    "Osteoarthritis": ("degenerative joint disease with cartilage loss, bone spurs, pain with activity", "Musculoskeletal System", ["Aging","Obesity","Joint Injury"], ["NSAIDs","Physical Therapy","Joint Replacement"]),
    "Rheumatoid Arthritis": ("chronic autoimmune symmetric polyarthritis with joint destruction", "Musculoskeletal System", ["Autoimmune (Anti-CCP)"], ["Methotrexate","Biologics","DMARDs"]),
    "Gout": ("crystal arthropathy from monosodium urate deposition, first MTP joint classically", "Musculoskeletal System", ["Hyperuricemia"], ["Colchicine","NSAIDs","Allopurinol","Febuxostat"]),
    "Osteoporosis": ("decreased bone mineral density with increased fracture risk, T-score ≤-2.5", "Musculoskeletal System", ["Estrogen Deficiency","Aging","Corticosteroids"], ["Bisphosphonates","Denosumab","Calcium","Vitamin D"]),
    "Systemic Lupus Erythematosus": ("multisystem autoimmune disease with anti-dsDNA antibodies, butterfly rash", "Musculoskeletal System", ["Autoimmune","Genetic"], ["Hydroxychloroquine","Corticosteroids","Immunosuppressants"]),
    "Ankylosing Spondylitis": ("chronic inflammatory arthritis of spine and SI joints, HLA-B27 associated", "Musculoskeletal System", ["HLA-B27","Genetic"], ["NSAIDs","TNF Inhibitors","Physical Therapy"]),
    "Osteomyelitis": ("bone infection, usually S. aureus, hematogenous or direct inoculation", "Musculoskeletal System", ["Staphylococcus aureus"], ["IV Antibiotics","Surgical Debridement"]),
    "Osteosarcoma": ("most common primary malignant bone tumor in children/adolescents, around knee", "Musculoskeletal System", ["Genetic (RB1/p53)"], ["Surgery","Chemotherapy"]),
    "Fibromyalgia": ("chronic widespread pain with tender points, fatigue, sleep disturbance", "Musculoskeletal System", ["Central Sensitization"], ["Duloxetine","Pregabalin","Exercise"]),
    "Paget Disease of Bone": ("disordered bone remodeling with enlarged deformed bones, elevated ALP", "Musculoskeletal System", ["Paramyxovirus Theory","Genetic"], ["Bisphosphonates"]),
    "Rhabdomyolysis": ("skeletal muscle breakdown releasing myoglobin, CK elevation, risk of AKI", "Musculoskeletal System", ["Trauma","Statins","Exertion"], ["IV Fluids","Alkalinization"]),
    "Duchenne Muscular Dystrophy": ("X-linked dystrophin deficiency causing progressive proximal muscle weakness, childhood onset", "Musculoskeletal System", ["DMD Gene Mutation"], ["Corticosteroids","Supportive Care"]),
    "Marfan Syndrome": ("autosomal dominant fibrillin-1 mutation, tall stature, aortic root dilation, lens subluxation", "Musculoskeletal System", ["FBN1 Gene Mutation"], ["Beta Blockers","Aortic Surveillance","Surgery"]),
    "Ehlers-Danlos Syndrome": ("connective tissue disorder with skin hyperelasticity and joint hypermobility", "Musculoskeletal System", ["Collagen Gene Mutations"], ["Supportive Care","Physical Therapy"]),
    "Herniated Disc": ("protrusion of nucleus pulposus through annulus fibrosus compressing nerve roots", "Musculoskeletal System", ["Degeneration","Trauma"], ["Physical Therapy","NSAIDs","Discectomy"]),
    "Rotator Cuff Tear": ("tear of supraspinatus/infraspinatus/subscapularis/teres minor tendons", "Musculoskeletal System", ["Degeneration","Trauma"], ["Physical Therapy","Surgical Repair"]),
    "ACL Tear": ("anterior cruciate ligament rupture from pivoting/deceleration injury", "Musculoskeletal System", ["Sports Injury"], ["ACL Reconstruction","Physical Therapy"]),
    "Meniscus Tear": ("tear of knee fibrocartilage causing locking, catching, and effusion", "Musculoskeletal System", ["Trauma","Degeneration"], ["Physical Therapy","Arthroscopic Repair"]),
    "Fracture (General)": ("break in bone continuity from trauma or pathologic process", "Musculoskeletal System", ["Trauma","Osteoporosis"], ["Reduction","Immobilization","Surgery"]),
    
    # Infectious Disease (50)
    "HIV/AIDS": ("HIV infection destroying CD4+ T cells leading to immunodeficiency and opportunistic infections", "Immune System", ["HIV Virus"], ["Antiretroviral Therapy"]),
    "Influenza": ("respiratory viral infection with fever, myalgia, cough, caused by influenza A/B", "Respiratory System", ["Influenza Virus"], ["Oseltamivir","Vaccination"]),
    "COVID-19": ("SARS-CoV-2 respiratory infection ranging from asymptomatic to ARDS and multiorgan failure", "Respiratory System", ["SARS-CoV-2"], ["Paxlovid","Remdesivir","Dexamethasone","Vaccination"]),
    "Malaria": ("protozoal infection by Plasmodium species transmitted by Anopheles mosquito, cyclic fevers", "Immune System", ["Plasmodium falciparum","Plasmodium vivax"], ["Chloroquine","Artemisinin","Atovaquone-Proguanil"]),
    "Sepsis": ("life-threatening organ dysfunction from dysregulated host response to infection", "Immune System", ["Bacterial Infection"], ["IV Antibiotics","IV Fluids","Vasopressors"]),
    "Cellulitis": ("spreading bacterial skin infection, usually S. aureus or Group A Strep", "Integumentary System", ["Staphylococcus aureus","Group A Streptococcus"], ["Antibiotics"]),
    "Pneumonia (Community-Acquired)": ("lung infection acquired outside hospital, S. pneumoniae most common typical cause", "Respiratory System", ["Streptococcus pneumoniae"], ["Antibiotics"]),
    "Urinary Tract Infection (Complicated)": ("UTI with complicating factors like obstruction or catheter", "Urinary System", ["E. coli"], ["Broad-Spectrum Antibiotics"]),
    "Osteomyelitis (Acute)": ("acute bone infection with fever and localized pain", "Musculoskeletal System", ["Staphylococcus aureus"], ["IV Antibiotics"]),
    "Endocarditis (Infective)": ("microbial infection of heart valve endothelium forming vegetations", "Cardiovascular System", ["S. aureus","S. viridans"], ["IV Antibiotics","Valve Surgery"]),
    "Meningitis (Bacterial)": ("acute bacterial infection of leptomeninges, medical emergency", "Nervous System", ["N. meningitidis","S. pneumoniae"], ["IV Antibiotics","Dexamethasone"]),
    "Herpes Simplex": ("HSV-1/2 infection causing oral/genital vesicular lesions with latency", "Integumentary System", ["HSV-1","HSV-2"], ["Acyclovir","Valacyclovir"]),
    "Varicella Zoster": ("primary infection (chickenpox) and reactivation (shingles) of VZV", "Integumentary System", ["Varicella Zoster Virus"], ["Acyclovir","Vaccination"]),
    "Hepatitis B (Chronic)": ("chronic HBV infection with risk of cirrhosis and HCC", "Digestive System", ["Hepatitis B Virus"], ["Tenofovir","Entecavir"]),
    "Clostridioides difficile Infection": ("toxin-producing colitis from C. difficile overgrowth after antibiotics", "Digestive System", ["Clostridioides difficile","Antibiotic Disruption"], ["Oral Vancomycin","Fidaxomicin","Fecal Transplant"]),
    "Candidiasis": ("fungal infection by Candida species, oral thrush, vaginal, or systemic", "Immune System", ["Candida albicans","Immunosuppression"], ["Fluconazole","Echinocandins"]),
    "Aspergillosis": ("fungal infection by Aspergillus, invasive form in immunocompromised, aspergilloma in cavities", "Respiratory System", ["Aspergillus fumigatus","Immunosuppression"], ["Voriconazole"]),
    "Lyme Disease": ("Borrelia burgdorferi infection transmitted by Ixodes tick, erythema migrans", "Immune System", ["Borrelia burgdorferi"], ["Doxycycline","Amoxicillin"]),
    "Syphilis": ("Treponema pallidum STI with stages: chancre, rash, gummas, neurosyphilis", "Reproductive System", ["Treponema pallidum"], ["Penicillin G"]),
    "Gonorrhea": ("Neisseria gonorrhoeae STI causing urethritis, cervicitis, PID", "Reproductive System", ["Neisseria gonorrhoeae"], ["Ceftriaxone"]),
    "Chlamydia": ("Chlamydia trachomatis STI, most common bacterial STI, often asymptomatic", "Reproductive System", ["Chlamydia trachomatis"], ["Azithromycin","Doxycycline"]),
    "Rabies": ("fatal viral encephalitis transmitted by animal bites, Negri bodies", "Nervous System", ["Rabies Virus"], ["Post-Exposure Prophylaxis","Vaccination"]),
    "Tetanus": ("Clostridium tetani toxin causing muscle rigidity and spasms", "Nervous System", ["Clostridium tetani Toxin"], ["Antitoxin","Metronidazole","Vaccination"]),
    "Botulism": ("Clostridium botulinum toxin causing descending flaccid paralysis", "Nervous System", ["Botulinum Toxin"], ["Antitoxin","Supportive Care"]),
    "Dengue": ("Aedes mosquito-transmitted flavivirus causing fever, rash, hemorrhagic complications", "Immune System", ["Dengue Virus"], ["Supportive Care"]),
    "Cholera": ("Vibrio cholerae infection causing profuse rice-water diarrhea and severe dehydration", "Digestive System", ["Vibrio cholerae"], ["Oral Rehydration","Antibiotics"]),
    "Typhoid Fever": ("Salmonella typhi systemic infection with stepwise fever, rose spots, hepatosplenomegaly", "Digestive System", ["Salmonella typhi"], ["Antibiotics","Vaccination"]),
    
    # Dermatologic (20)
    "Psoriasis": ("chronic autoimmune skin disease with silvery scaly plaques, T-cell mediated", "Integumentary System", ["T-Cell Mediated Autoimmune"], ["Topical Steroids","Methotrexate","Biologics","Phototherapy"]),
    "Eczema": ("atopic dermatitis, chronic pruritic inflammatory skin condition, often in atopic triad", "Integumentary System", ["Filaggrin Mutations","Immune Dysregulation"], ["Emollients","Topical Steroids","Calcineurin Inhibitors"]),
    "Acne Vulgaris": ("inflammatory skin condition with comedones, papules, pustules from pilosebaceous unit", "Integumentary System", ["Excess Sebum","P. acnes","Androgen Effect"], ["Retinoids","Benzoyl Peroxide","Antibiotics","Isotretinoin"]),
    "Basal Cell Carcinoma": ("most common skin cancer, locally invasive, rarely metastatic, pearly papule", "Integumentary System", ["UV Exposure"], ["Surgical Excision","Mohs Surgery"]),
    "Squamous Cell Carcinoma (Skin)": ("second most common skin cancer, can metastasize, keratinizing tumor", "Integumentary System", ["UV Exposure","Immunosuppression"], ["Surgical Excision","Radiation"]),
    "Contact Dermatitis": ("inflammatory skin reaction from allergen (type IV) or irritant exposure", "Integumentary System", ["Allergen/Irritant Exposure"], ["Avoidance","Topical Steroids"]),
    "Urticaria": ("hives, transient pruritic wheals from mast cell degranulation", "Integumentary System", ["Allergens","Infections","Autoimmune"], ["Antihistamines"]),
    "Pemphigus Vulgaris": ("autoimmune blistering disease with IgG against desmoglein, flaccid blisters", "Integumentary System", ["Anti-Desmoglein Antibodies"], ["Corticosteroids","Rituximab"]),
    "Bullous Pemphigoid": ("autoimmune subepidermal blistering, tense blisters, anti-BP180/BP230 antibodies", "Integumentary System", ["Anti-BP180 Antibodies"], ["Topical/Systemic Steroids"]),
    "Stevens-Johnson Syndrome": ("severe mucocutaneous reaction usually drug-induced, <10% BSA detachment", "Integumentary System", ["Drug Reaction"], ["Stop Offending Drug","Supportive Care"]),
    "Vitiligo": ("autoimmune destruction of melanocytes causing depigmented patches", "Integumentary System", ["Autoimmune Melanocyte Destruction"], ["Topical Steroids","Phototherapy","JAK Inhibitors"]),
    "Rosacea": ("chronic facial erythema with telangiectasia, papules, and rhinophyma", "Integumentary System", ["Unknown","Demodex"], ["Topical Metronidazole","Brimonidine","Doxycycline"]),
    
    # Psychiatric (20)
    "Major Depressive Disorder": ("persistent depressed mood or anhedonia with neurovegetative symptoms for 2+ weeks", "Nervous System", ["Serotonin/NE Dysregulation","Genetic","Psychosocial"], ["SSRIs","SNRIs","CBT","ECT"]),
    "Bipolar Disorder": ("mood disorder with manic and depressive episodes", "Nervous System", ["Genetic","Neurochemical"], ["Lithium","Valproate","Quetiapine","Lamotrigine"]),
    "Schizophrenia": ("chronic psychotic disorder with positive (hallucinations, delusions) and negative symptoms", "Nervous System", ["Dopamine Excess","Genetic"], ["Antipsychotics"]),
    "Generalized Anxiety Disorder": ("excessive persistent worry about multiple domains for 6+ months", "Nervous System", ["GABA/Serotonin Dysregulation"], ["SSRIs","Buspirone","CBT"]),
    "Panic Disorder": ("recurrent unexpected panic attacks with fear of future attacks", "Nervous System", ["Autonomic Dysregulation"], ["SSRIs","CBT"]),
    "Obsessive-Compulsive Disorder": ("recurrent intrusive thoughts (obsessions) and repetitive behaviors (compulsions)", "Nervous System", ["Serotonin Dysregulation","Cortico-Striatal Circuit"], ["SSRIs","CBT","ERP"]),
    "PTSD": ("post-traumatic stress disorder with re-experiencing, avoidance, hyperarousal after trauma", "Nervous System", ["Traumatic Event"], ["SSRIs","CPT","EMDR"]),
    "ADHD": ("attention deficit hyperactivity disorder with inattention and/or hyperactivity-impulsivity", "Nervous System", ["Dopamine/NE Dysregulation","Genetic"], ["Methylphenidate","Amphetamines","Atomoxetine"]),
    "Autism Spectrum Disorder": ("neurodevelopmental disorder with social communication deficits and restricted behaviors", "Nervous System", ["Genetic","Neurodevelopmental"], ["Behavioral Therapy","Supportive Care"]),
    "Anorexia Nervosa": ("eating disorder with restriction, low body weight, and fear of weight gain", "Nervous System", ["Psychological","Genetic"], ["Nutritional Rehabilitation","Psychotherapy"]),
    "Bulimia Nervosa": ("eating disorder with binge eating and compensatory purging", "Nervous System", ["Psychological","Serotonin Dysregulation"], ["CBT","SSRIs"]),
    "Substance Use Disorder": ("pathological pattern of substance use causing impairment, tolerance, withdrawal", "Nervous System", ["Dopamine Reward Pathway","Genetic","Environmental"], ["Behavioral Therapy","MAT"]),
    "Alcohol Use Disorder": ("problematic alcohol use with tolerance, withdrawal, and impaired control", "Nervous System", ["Genetic","GABA/Glutamate Dysregulation"], ["Naltrexone","Acamprosate","Disulfiram","CBT"]),
    
    # Ophthalmologic (10)
    "Glaucoma": ("optic neuropathy with increased intraocular pressure, progressive visual field loss", "Nervous System", ["Elevated IOP","Optic Nerve Damage"], ["Prostaglandin Analogs","Beta Blockers","Trabeculectomy"]),
    "Cataracts": ("opacity of crystalline lens causing progressive vision loss", "Nervous System", ["Aging","UV Exposure","Diabetes"], ["Cataract Surgery with IOL"]),
    "Macular Degeneration": ("age-related degeneration of macula causing central vision loss", "Nervous System", ["Aging","Genetic"], ["Anti-VEGF Injections","AREDS Vitamins"]),
    "Retinal Detachment": ("separation of neurosensory retina from RPE, flashes and floaters", "Nervous System", ["Vitreous Traction","Myopia"], ["Laser Photocoagulation","Vitrectomy","Scleral Buckle"]),
    "Diabetic Retinopathy": ("retinal microvascular damage from chronic hyperglycemia, leading cause of blindness in working age", "Nervous System", ["Diabetes"], ["Glycemic Control","Anti-VEGF","Laser Photocoagulation"]),
    
    # Pediatric (15)
    "Kawasaki Disease": ("acute vasculitis of childhood affecting coronary arteries, fever >5 days", "Cardiovascular System", ["Unknown","Infectious Trigger"], ["IVIG","Aspirin"]),
    "Croup": ("viral laryngotracheobronchitis with barking cough and stridor in children", "Respiratory System", ["Parainfluenza Virus"], ["Dexamethasone","Nebulized Epinephrine"]),
    "Intussusception": ("telescoping of intestine into adjacent segment, target sign on ultrasound", "Digestive System", ["Lead Point","Viral"], ["Air Enema Reduction","Surgery"]),
    "Pyloric Stenosis": ("hypertrophic pyloric muscle causing projectile vomiting in infants (2-8 weeks)", "Digestive System", ["Pyloric Muscle Hypertrophy"], ["Pyloromyotomy"]),
    "Hirschsprung Disease": ("congenital absence of ganglion cells in distal colon causing functional obstruction", "Digestive System", ["RET Gene Mutation"], ["Pull-Through Surgery"]),
    "Neonatal Respiratory Distress Syndrome": ("surfactant deficiency in premature infants causing alveolar collapse", "Respiratory System", ["Prematurity","Surfactant Deficiency"], ["Exogenous Surfactant","CPAP"]),
    "Phenylketonuria": ("autosomal recessive phenylalanine hydroxylase deficiency, intellectual disability if untreated", "Endocrine System", ["PAH Gene Mutation"], ["Low-Phenylalanine Diet"]),
    "Down Syndrome": ("trisomy 21 causing intellectual disability, characteristic facies, heart defects", "Nervous System", ["Trisomy 21"], ["Supportive Care","Early Intervention"]),
    "Turner Syndrome": ("45,X monosomy causing short stature, webbed neck, ovarian dysgenesis in females", "Endocrine System", ["X Monosomy"], ["Growth Hormone","Estrogen Replacement"]),
    "Klinefelter Syndrome": ("47,XXY causing tall stature, gynecomastia, small testes, infertility in males", "Endocrine System", ["XXY Karyotype"], ["Testosterone Replacement"]),
    "Congenital Heart Disease": ("structural heart defects present at birth, most common birth defect category", "Cardiovascular System", ["Genetic","Environmental"], ["Surgery","Catheter Intervention"]),
    "Febrile Seizure": ("seizure in children 6mo-5yr associated with fever, usually benign", "Nervous System", ["Fever"], ["Treat Fever","Reassurance"]),
    "Nephroblastoma": ("Wilms tumor, most common renal malignancy in children, flank mass", "Urinary System", ["WT1/WT2 Mutations"], ["Nephrectomy","Chemotherapy"]),
    
    # Additional diseases
    "Septic Arthritis": ("joint infection, medical emergency, usually S. aureus", "Musculoskeletal System", ["Staphylococcus aureus"], ["Joint Drainage","IV Antibiotics"]),
    "Necrotizing Fasciitis": ("rapidly spreading deep soft tissue infection, surgical emergency", "Integumentary System", ["Group A Strep","Polymicrobial"], ["Surgical Debridement","IV Antibiotics"]),
    "Toxic Shock Syndrome": ("toxin-mediated shock from S. aureus or Group A Strep", "Immune System", ["Staphylococcal/Streptococcal Toxin"], ["Antibiotics","Supportive Care"]),
    "Anaphylaxis": ("severe type I hypersensitivity with cardiovascular collapse and bronchospasm", "Immune System", ["IgE-Mediated Allergic Reaction"], ["Epinephrine","Antihistamines","Steroids"]),
    "Sarcoidosis (Systemic)": ("multisystem granulomatous disease, bilateral hilar LAD, elevated ACE", "Immune System", ["Unknown"], ["Corticosteroids"]),
    "Amyloidosis": ("extracellular deposition of misfolded proteins causing organ dysfunction", "Immune System", ["Misfolded Protein Deposition"], ["Chemotherapy","Organ Transplant"]),
    "Hemolytic Uremic Syndrome": ("triad of microangiopathic hemolytic anemia, thrombocytopenia, and AKI, often post-E. coli O157:H7", "Immune System", ["Shiga Toxin","E. coli O157:H7"], ["Supportive Care","Eculizumab"]),
    "Goodpasture Syndrome": ("anti-GBM disease causing rapidly progressive GN and pulmonary hemorrhage", "Urinary System", ["Anti-GBM Antibodies"], ["Plasmapheresis","Immunosuppression"]),
    "Granulomatosis with Polyangiitis": ("ANCA-associated vasculitis (formerly Wegener's), affects lungs, kidneys, sinuses", "Immune System", ["c-ANCA (Anti-PR