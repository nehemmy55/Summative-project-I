# Error analysis for `full_ft_rag` (Small, full FT, + MedQuAD + retrieval)

## Multiple-choice questions (n=373, accuracy 24.9%)


**MCQ accuracy by negation in the question (except / not / false ...)**

| negated   |   n |   accuracy |
|:----------|----:|-----------:|
| False     | 260 |       23.5 |
| True      | 113 |       28.3 |


**MCQ accuracy by question length**

| q_words                       |   n |   accuracy |
|:------------------------------|----:|-----------:|
| short (<=10 words)            | 134 |       29.1 |
| medium (11-25)                | 150 |       21.3 |
| long (>25, clinical vignette) |  89 |       24.7 |


**MCQ accuracy by length of the correct option**

| answer_words   |   n |   accuracy |
|:---------------|----:|-----------:|
| 1-2 words      | 168 |       28.6 |
| 3-5 words      | 112 |       25.9 |
| 6+ words       |  93 |       17.2 |


**MCQ accuracy by country of the contributor**

| country   |   n |   accuracy |
|:----------|----:|-----------:|
| GH        | 162 |       25.9 |
| KE        |  53 |       18.9 |
| MW        |  24 |       16.7 |
| NG        | 128 |       27.3 |
| TZ        |   1 |        0   |
| ZA        |   5 |       40   |


**MCQ accuracy by specialty (8 largest)**

| specialty                 |   n |   accuracy |
|:--------------------------|----:|-----------:|
| Psychiatry                |  17 |       11.8 |
| General_Surgery           |  59 |       16.9 |
| Pathology                 |  23 |       21.7 |
| Infectious_Disease        |  19 |       26.3 |
| Neurology                 |  15 |       26.7 |
| Obstetrics_and_Gynecology |  62 |       27.4 |
| Pediatrics                |  47 |       27.7 |
| Cardiology                |  17 |       29.4 |


**Position of the chosen option**

|          |   correct option (%) |   model choice (%) |
|:---------|---------------------:|-------------------:|
| no match |                  0   |                1.6 |
| A        |                 20.9 |               20.4 |
| B        |                 24.1 |               19.8 |
| C        |                 20.4 |               27.9 |
| D        |                 19   |               18.2 |
| E        |                 15.5 |               12.1 |


Questions with an 'all/none of the above' option: n=53, accuracy 20.8%


### Correct MCQ examples

- **question:** 5-week-old child is rushed to the ER with cough, fast breathing, tachycardia and poor feeding. Abdominal exam reveals tender enlarged liver. What is the possible cause?
- **options:** ['\u200bSepticaemia', 'galactosemia', 'pneumonia', 'congestive cardiac failure', 'n/a']
- **answer:** congestive cardiac failure
- **prediction:** congestive cardiac failure

- **question:** Features of catatonia includes all of the following except
- **options:** ['Catalepsy', 'Mutism', 'Stupor', 'Cataplexy', 'Echolalia']
- **answer:** Cataplexy
- **prediction:** Cataplexy

- **question:** Microorganisms that are implicated in the pathogenesis of rheumatoid arthritis include except
- **options:** ['Mycoplasma', 'Papovirus B20', 'Mycobacterium', 'Retrovirus', 'n/a']
- **answer:** Papovirus B20
- **prediction:** Papovirus B20

- **question:** An infant is delivered at 27/40 and taken to the neonatal unit. Which problem is most likely to be experienced?
- **options:** ['Incomplete formation of the epidermis from the mesoderm', 'Excessive vernix formation', 'Lanugo shedding', 'Thermoregulation due to thin skin', 'Lack of hair follicle development']
- **answer:** Thermoregulation due to thin skin
- **prediction:** Thermoregulation due to thin skin

- **question:** A 3-day old neonate delivered at term presents with oozing of blood from the umbilical cord stump and passage of dark stools of six hours duration. He had stopped sucking and became breathless just before presentation. Which of the following statements is valid?
- **options:** ['The diagnosis is haemolytic disease of newborn', 'Intramuscular Vitamin K 1mg is recommended', 'Double volume exchange transfusion is indicated', 'Tender hepatomegaly will be expected', 'Apt test is important']
- **answer:** Tender hepatomegaly will be expected
- **prediction:** Tender hepatomegaly will be expected


### Incorrect MCQ examples

- **question:** Which of the following best describes Magnesium sulphate
- **options:** ['Loss of deep tendon reflexes before respiratory depression with regards to toxicity', 'It adequately prevents eclampsia in preeclampsia without severe features', 'Is mainly excreted by the kidney and conjugation in the liver', 'Provides neuroprotection for the fetus before 34 weeks of gestation', 'Overdose is infrequently treated with 10ml of 10% calcium gluconate']
- **answer:** Loss of deep tendon reflexes before respiratory depression with regards to toxicity
- **prediction:** Is mainly excreted by the kidney and conjugation in the liver
- **specialty:** Obstetrics_and_Gynecology

- **question:** A 32-year-old Para 2+1 presents with one year history amenorrhea following evacuation of the uterus due to septic incomplete abortion. Her hormonal profile is normal. What is the likely diagnosis
- **options:** ['Premature menopause', 'latrogenic ovarian failure', 'Hypothalamic pituitary amenorrhea', "Sheehan's syndrome", "Asherman's syndrome"]
- **answer:** Asherman's syndrome
- **prediction:** Premature menopause
- **specialty:** Obstetrics_and_Gynecology

- **question:** Which of the following is a component of the Alvarado’s score
- **options:** ['periumbilical pain', 'Anorexia', 'Diarrhoea', 'Leucopenia', 'n/a']
- **answer:** Anorexia
- **prediction:** Periumbilical pain
- **specialty:** General_Surgery

- **question:** The overall goal of the primary survey is to:
- **options:** ['n/a', 'Identify and treat limb-threatening injuries', 'Identify those patients in whom further care is futile', 'Identify and treat life-threatening conditions in trauma', 'Identify all traumatic injuries of the patient']
- **answer:** Identify and treat life-threatening conditions in trauma
- **prediction:** Identify and treat limb-threatening injuries
- **specialty:** Emergency_Medicine

- **question:** A 29 years old 40 weeks gestation reports to labour ward with maternal exhaustion. Examining her cervix to be 3 cm dilated and partially effaced. A decision to induce her. How would you proceed?
- **options:** ['Insert a forleys catheter extra amniotic under traction then observe for labour', 'Start oxytocin drip', 'Perform artificial rupture of membranes and if clear start oxytocin drop', 'Insert prostaglandin pessary then perform artificial rupture of membranes', 'Insert misoprostol 25mg into posterior fornix then review after 4 hours']
- **answer:** Insert misoprostol 25mg into posterior fornix then review after 4 hours
- **prediction:** Insert prostaglandin pessary then perform artificial rupture of membranes
- **specialty:** Obstetrics_and_Gynecology

- **question:** What is the most common cause of malnutrition in children in Africa?
- **options:** ['Lack of access to food', 'Poor sanitation', 'Infectious diseases', 'Lack of education', 'Political instability']
- **answer:** Infectious diseases
- **prediction:** Political instability
- **specialty:** Pediatrics

- **question:** The following are contraindications to external cephalic version except:
- **options:** ['Previous Caesarean section Scar', 'Twin pregnancy', 'Placenta previa', 'Membrane rupture', 'Complete breech position']
- **answer:** Complete breech position
- **prediction:** Twin pregnancy
- **specialty:** Obstetrics_and_Gynecology

- **question:** A man was found to have collapsed at work and was brought in dead. On autopsy, the pathologist noticed a raindrop pattern of hyperpigmentation on the skin. This finding raises suspicion of:
- **options:** ['Death by strangulation', 'Chronic arsenic poisoning', 'Addison’s disease', 'Chloasma', 'N/A']
- **answer:** Chronic arsenic poisoning
- **prediction:** Death by strangulation
- **specialty:** Pathology


## Open questions (n=323)

**Automatic error categories**

| error_type                 |   afrimedqa |   medquad |
|:---------------------------|------------:|----------:|
| good (ROUGE-L >= 0.5)      |          28 |        32 |
| partially correct          |          26 |        44 |
| repetitive                 |           0 |         1 |
| restates the question      |          28 |         1 |
| wrong or unrelated content |          41 |       122 |


### Best open answers

- **source:** medquad
- **question:** Who is at risk for Hypopharyngeal Cancer? ?
- **answer:** Use of tobacco products and heavy drinking can affect the risk of developing hypopharyngeal cancer. Anything that increases your risk of getting a disease is called a risk factor.
- **prediction:** Use of tobacco products and heavy drinking can affect the risk of developing hypopharyngeal cancer. Anything that increases your risk of getting a disease is called a risk factor.
- **rougeL:** 1.0

- **source:** medquad
- **question:** Who is at risk for Pancreatic Neuroendocrine Tumors (Islet Cell Tumors)? ?
- **answer:** Having certain syndromes can increase the risk of pancreatic NETs. Anything that increases your risk of getting a disease is called a risk factor.
- **prediction:** Having certain syndromes can increase the risk of pancreatic NETs. Anything that increases your risk of getting a disease is called a risk factor.
- **rougeL:** 1.0

- **source:** medquad
- **question:** Who is at risk for Adult Primary Liver Cancer? ?
- **answer:** Having hepatitis or cirrhosis can affect the risk of adult primary liver cancer. Anything that increases your chance of getting a disease is called a risk factor.
- **prediction:** Having hepatitis or cirrhosis can affect the risk of adult primary liver cancer. Anything that increases your chance of getting a disease is called a risk factor.
- **rougeL:** 1.0

- **source:** medquad
- **question:** Who is at risk for Adult Soft Tissue Sarcoma? ?
- **answer:** Having certain inherited disorders can increase the risk of adult soft tissue sarcoma. Anything that increases your risk of getting a disease is called a risk factor.
- **prediction:** Having certain inherited disorders can increase the risk of adult soft tissue sarcoma. Anything that increases your risk of getting a disease is called a risk factor.
- **rougeL:** 1.0

- **source:** medquad
- **question:** Who is at risk for Childhood Soft Tissue Sarcoma? ?
- **answer:** Having certain diseases and inherited disorders can increase the risk of childhood soft tissue sarcoma. Anything that increases your risk of getting a disease is called a risk factor.
- **prediction:** Having certain diseases and inherited disorders can increase the risk of childhood soft tissue sarcoma. Anything that increases your risk of getting a disease is called a risk factor.
- **rougeL:** 1.0


### Open answers: wrong or unrelated content

- **source:** afrimedqa
- **question:** Write short notes on immune reconstitution inflammatory syndrome
- **answer:** Immune reconstitution inflammatory syndrome (IRIS) is defined as: paradoxical clinical deterioration after starting HAART, resulting from improving immune system interaction with organisms that have colonised the body during the early stages of HIV infection. A wide range of pathogens causes IRIS.
- **prediction:** Immunoglobulins : mainly IgA, mainly igA.
- **rougeL:** 0.0

- **source:** medquad
- **question:** Do you have information about Limb Loss
- **answer:** Summary : People can lose all or part of an arm or leg for a number of reasons. Common ones include - Problems with blood circulation. These may be the result of atherosclerosis or diabetes. Severe cases may result in amputation.
- **prediction:** Summary : Limbloss is a disease in which the body's cells form in the tissues of the uterus. It is caused by a combination of factors such as - the size of the abdomen, the
- **rougeL:** 0.1643835616438356

- **source:** medquad
- **question:** What are the treatments for Dumping Syndrome ?
- **answer:** Treatment for dumping syndrome includes changes in eating, diet, and nutrition; medication; and, in some cases, surgery. Many people with dumping syndrome have mild symptoms that improve over time with simple dietary changes.
- **prediction:** There is no cure for dumping syndrome. Treatment is symptomatic and supportive.
- **rougeL:** 0.1777777777777777

- **source:** medquad
- **question:** How to prevent Childhood Interstitial Lung Disease ?
- **answer:** At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A counselor can explain the risk of children inheriting chILD.
- **prediction:** To prevent childhood interstitial lung disease (chILD), a health care provider may recommend a physical exam, a medical history, and a blood test.
- **rougeL:** 0.2424242424242424


### Open answers: restates the question

- **source:** afrimedqa
- **question:** Discuss the clinical features and management of malaria in pediatric patients in Nigeria
- **answer:** Clinical features of malaria in pediatric patients can vary depending on factors such as age, immunity, and malaria species. Common clinical features include: 1. **Fever**: Persistent fever is often the first and most common symptom of malaria in children.
- **prediction:** Clinical features and management of malaria in pediatric patients in Nigeria
- **rougeL:** 0.32

- **source:** afrimedqa
- **question:** How is bronchopneumonia managed in Nigeria tertiary hospitals
- **answer:** Management of bronchopneumonia in Nigerian tertiary hospitals typically involves: 1. _Oxygen therapy_: Supplemental oxygen to maintain adequate oxygen saturation. 2. _Antibiotics_: Broad-spectrum antibiotics, such as ceftriaxone, cefotaxime, or ampicillin, to cover common bacterial pathogens. 3.
- **prediction:** Bronchopneumonia is managed in Nigeria tertiary hospitals.
- **rougeL:** 0.1860465116279069

- **source:** afrimedqa
- **question:** A 34-year-old patient presents to the emergency department with a history of rectal bleeding that is bright red and only noticed on wiping. The patient was complaining of constipation with no history of weight loss. The patient is hemodynamically stable with normal blood results. List FOUR possible causes of the bleeding?
- **answer:** Bleeding hemorrhoids. Anal fissure Rectal polyp Rectal/anal carcinoma Ulcerative colitis Diverticular disease
- **prediction:** Causes of the bleeding include constipation with no history of weight loss.
- **rougeL:** 0.08

- **source:** afrimedqa
- **question:** What is the leading cause of death among children under 5 years old in sub-Saharan Africa?
- **answer:** Malaria. Malaria is the biggest killer of young children under the age of 5 in the sub-Saharan Africa region.
- **prediction:** Cataract, a clouding of the eye's lens, is the leading cause of death among children under 5 years old in sub-Saharan Africa.
- **rougeL:** 0.4545454545454545


### Open answers: repetitive

- **source:** medquad
- **question:** What are the symptoms of Respiratory Failure ?
- **answer:** The signs and symptoms of respiratory failure depend on its underlying cause and the levels of oxygen and carbon dioxide in the blood.
- **prediction:** Signs and symptoms of respiratory failure include fatigue, fatigue, and fatigue. These and other signs and symptoms may be caused by respiratory failure or by other conditions.
- **rougeL:** 0.3199999999999999


### Open answers: partially correct

- **source:** medquad
- **question:** Do you have information about Palliative Care
- **answer:** Summary : Palliative care is treatment of the discomfort, symptoms, and stress of serious illness.
- **prediction:** Summary : Palliative care is a type of care that helps people with chronic pain recover from their injuries. It can help people with a chronic pain reliever, such as aches, pains, and pains
- **rougeL:** 0.2553191489361702

- **source:** afrimedqa
- **question:** What are the challenges in diagnosing and managing nasopharyngeal stenosis in African populations?
- **answer:** Challenges include limited access to diagnostic imaging and otolaryngologists with expertise in managing this condition.
- **prediction:** Challenges include late presentation, limited access to nasopharyngeal stenosis therapy, and lack of awareness about the disease.
- **rougeL:** 0.375

- **source:** medquad
- **question:** What is (are) Familial Periodic Paralyses ?
- **answer:** Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affected muscles become slack, weak, and unable to contract.
- **prediction:** Familial periodic paralyses are a group of neuromuscular disorders that affect the brain and spinal cord.
- **rougeL:** 0.3859649122807017

- **source:** medquad
- **question:** Do you have information about Health Insurance
- **answer:** Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when you need medical care.
- **prediction:** Summary : Health insurance helps pay for medical care, including the cost of diabetes care. Health insurance options include private health insurance and government health insurance.
- **rougeL:** 0.2686567164179104

## Is MCQ accuracy above chance?

One-sided binomial test against 20% (five options) on the 373 test MCQs.

| System | correct / n | accuracy | p-value |
|---|---|---|---|
| FLAN-T5-small zero-shot (baseline) | 72 / 373 | 19.3% | 0.652 |
| Small, LoRA r=8, AfriMed | 88 / 373 | 23.6% | 0.050 |
| Small, LoRA r=32, AfriMed | 86 / 373 | 23.1% | 0.081 |
| Small, full FT, AfriMed | 85 / 373 | 22.8% | 0.101 |
| Small, full FT, + MedQuAD | 82 / 373 | 22.0% | 0.185 |
| Small, full FT, + MedQuAD + retrieval | 93 / 373 | 24.9% | 0.012 |
| Base, full FT, AfriMed | 76 / 373 | 20.4% | 0.449 |
| Base, full FT, + MedQuAD + retrieval | 84 / 373 | 22.5% | 0.125 |

Across the 7 fine-tuned models, 43.2% of the questions are answered correctly by none and 2.4% by all; if the models were guessing independently these would be about 21.0% and 0.001%.


## When does retrieval help? (open questions, ROUGE-L by TF-IDF similarity of the retrieved entry)

|                          |   with retrieval |   without retrieval |   TF-IDF answer |   context/answer F1 |   output copies context |   n |
|:-------------------------|-----------------:|--------------------:|----------------:|--------------------:|------------------------:|----:|
| ('afrimedqa', '< 0.2')   |             13   |                19.3 |             7.9 |                12.3 |                    14.3 |   7 |
| ('afrimedqa', '0.2-0.4') |             28.2 |                27.3 |            19.7 |                23.5 |                    18.4 |  87 |
| ('afrimedqa', '0.4-0.6') |             43.5 |                38.1 |            34.6 |                39.2 |                    52.2 |  23 |
| ('afrimedqa', '>= 0.6')  |             26.6 |                26.7 |            29.9 |                31.3 |                    16.7 |   6 |
| ('medquad', '< 0.2')     |             17.1 |                18.4 |             7.1 |                 9   |                     0   |  12 |
| ('medquad', '0.2-0.4')   |             21.6 |                20.2 |            15.7 |                18.6 |                     2.5 |  80 |
| ('medquad', '0.4-0.6')   |             32.5 |                26.1 |            17   |                20.2 |                     2.4 |  82 |
| ('medquad', '>= 0.6')    |             48.4 |                35.9 |            26   |                28.1 |                     0   |  26 |


## Possible causes attached to the wrong answers

543 of 696 test answers are wrong (MCQ: wrong option; open: ROUGE-L < 0.5). One answer can have several tags or none; the tags are rule-based hints. The full table is in `results/error_table.csv`.

| possible_cause                             |   wrong answers |
|:-------------------------------------------|----------------:|
| nothing similar in the knowledge base      |              89 |
| negated question                           |              81 |
| picked option C (position bias)            |              79 |
| long correct option                        |              77 |
| 'all/none of the above' option             |              42 |
| instruction needs a long structured answer |              15 |
