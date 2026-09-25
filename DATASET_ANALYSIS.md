# Dataset Exploration Analysis
Welcome to your first hackathon! Below are the insights derived from exploring the dataset.
## 1. File Statistics (Rows, Columns, and Missing Values)
### train_ground_truth.tsv
- **Rows**: 2,206,821
- **Columns**: 2
- **Missing Values per Column**:
  - `source1_entity_id`: 0 missing
  - `matched_entity_ids`: 123,247 missing

### train_source1.tsv
- **Rows**: 2,206,821
- **Columns**: 4
- **Missing Values per Column**:
  - `entity_id`: 0 missing
  - `business_name`: 0 missing
  - `business_address`: 0 missing
  - `country`: 0 missing

### train_source2.tsv
- **Rows**: 5,034,616
- **Columns**: 4
- **Missing Values per Column**:
  - `entity_id`: 0 missing
  - `business_name`: 2 missing
  - `business_address`: 168,967 missing
  - `country`: 0 missing

### train_source3.tsv
- **Rows**: 5,285,603
- **Columns**: 4
- **Missing Values per Column**:
  - `entity_id`: 0 missing
  - `business_name`: 13 missing
  - `business_address`: 175,916 missing
  - `country`: 0 missing

### test_source1.tsv
- **Rows**: 1,732,544
- **Columns**: 4
- **Missing Values per Column**:
  - `entity_id`: 0 missing
  - `business_name`: 0 missing
  - `business_address`: 0 missing
  - `country`: 0 missing

### test_source2.tsv
- **Rows**: 4,887,273
- **Columns**: 4
- **Missing Values per Column**:
  - `entity_id`: 0 missing
  - `business_name`: 46 missing
  - `business_address`: 129,408 missing
  - `country`: 0 missing

### test_source3.tsv
- **Rows**: 5,082,316
- **Columns**: 4
- **Missing Values per Column**:
  - `entity_id`: 0 missing
  - `business_name`: 59 missing
  - `business_address`: 136,098 missing
  - `country`: 0 missing

## 2. Match / Class Distribution
How many Source 2/3 records match each Source 1 business in the training set?
- **0 Matches**: 123,247
- **1 Match**: 119,157
- **2 to 5 Matches**: 1,712,125
- **More than 5 Matches**: 252,292

## 3. 'France' Country Check
- **Confirmed**: 'France' does NOT appear in any training file.
- **Confirmed**: 'France' appears in the test data.

## 4. Leakage Check
- **Confirmed**: No ground-truth or label columns (`matched_entity_ids`) exist in the test files.

## 5. Real Example Pairs (Noise Inspection)
Here are some real examples from the dataset showing a Source 1 business alongside its matched Source 2/Source 3 records:

### Example 1
**Source 1 Reference:**
- **Name**: `Maure Williams Colombier Inc`
- **Address**: `85 Wayne Avenue, Ticonderoga, NY`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `Maure Wilblims Colombier Inc`
- [Source 2] **Address**: `nan`
- [Source 2] **Name**: `Maure Williams Colombier`
- [Source 2] **Address**: `nan`
- [Source 3] **Name**: `Dréxkor`
- [Source 3] **Address**: `85 Wanye Avenue, Ticonderoga Townshiip, New York`
- [Source 3] **Name**: `maurewilliamscolombier.com`
- [Source 3] **Address**: `Wayne Ave, Ticonderoga Townshiip, New York`
- [Source 3] **Name**: `Maure Williams Inc Center`
- [Source 3] **Address**: `nan`

---

### Example 2
**Source 1 Reference:**
- **Name**: `Raj Investments LLP`
- **Address**: `6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`
- [Source 2] **Address**: `6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu`
- [Source 2] **Name**: `Raj Investments LLP`
- [Source 2] **Address**: `6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu`
- [Source 3] **Name**: `Raj Investments எல்எல்பி`
- [Source 3] **Address**: `6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN`
- [Source 3] **Name**: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`
- [Source 3] **Address**: `6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு`

---

### Example 3
**Source 1 Reference:**
- **Name**: `Dahlia Power Reliable Scientific LLC`
- **Address**: `630 45th Terrace, Kansas City, MO`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `Dahlia Power Reliable`
- [Source 2] **Address**: `KANSAS CITY, MO, 630 45ND TERRACE, null`
- [Source 2] **Name**: `Dahlia Power Reliable Scientific`
- [Source 2] **Address**: `45ND TERRACE, null, KANSAS CITY, MO`
- [Source 3] **Name**: `Dahlia Ponr Reliable Scientific LLC`
- [Source 3] **Address**: `Missouri, 630 45th Terrace, Kansas City`

---

### Example 4
**Source 1 Reference:**
- **Name**: `Ss Food Private Limited`
- **Address**: `Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `एसएस फूड प्राइवेट लिमिटेड`
- [Source 2] **Address**: `AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203, उत्तर प्रदेश`
- [Source 2] **Name**: `एसएस फूड प्राइवेट लिमिटेड`
- [Source 2] **Address**: `AF-0684, Uttar Pradesh, GHAZIABAD, 9487203`
- [Source 3] **Name**: `एसएस फूड प्राइवेट लिमिटेड`
- [Source 3] **Address**: `Af-684, Ghaziabad, UP`

---

### Example 5
**Source 1 Reference:**
- **Name**: `Payne Enterprises`
- **Address**: `3315 Fremont Street, Peoria, IL`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `Payne Énterprises`
- [Source 2] **Address**: `3315 FREMONT ST, PEORIA, IL`
- [Source 2] **Name**: `Payne Enterpires`
- [Source 2] **Address**: `3315 FREMONT ST, PEORIA, IL`
- [Source 2] **Name**: `PAYNE-ENRTPRMISES`
- [Source 2] **Address**: `3315 FREMONT SAINT, PEORIA, IL`
- [Source 3] **Name**: `Payne Etrepndiels`
- [Source 3] **Address**: `3315 Fremont St, Peoria, Illinois`
- [Source 3] **Name**: `Payne Énterprises`
- [Source 3] **Address**: `3315 Fremont Street, Peoria, Illinois`
- [Source 3] **Name**: `Payne Enterprises  LLC`
- [Source 3] **Address**: `Fremont St, Peoria, Illinois`

---

### Example 6
**Source 1 Reference:**
- **Name**: `Lumay Boral`
- **Address**: `1056 Belden Avenue, Akron, OH`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `Lumay Boral Inc.`
- [Source 2] **Address**: `1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH`
- [Source 3] **Name**: `Lumay Bóral`
- [Source 3] **Address**: `1056c Belden Ave, AKON, Ohio`
- [Source 3] **Name**: `Lumay Boral`
- [Source 3] **Address**: `1056c Belden Ave, AKON, Ohio`
- [Source 3] **Name**: `Lumay Bóral`
- [Source 3] **Address**: `1056c Belden Avenue, AKON, Ohio`

---

### Example 7
**Source 1 Reference:**
- **Name**: `Red Ventures Private Limited`
- **Address**: `Rajasthan, Jaipur, Banipark, Gokul Apartment, E-3A Kanti Chandra Road, G-1`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `रेड वेंचर्स प्राइवेट लिमिटेड`
- [Source 2] **Address**: `G-1, BANIPARK, JAIPUR, Rajasthan`
- [Source 3] **Name**: `Red Ventures Private`
- [Source 3] **Address**: `Doro No 316 G-1, Gokul Apartment, E-3a Kanti Chandra Road, Banipark, Subhash Nagar, RJ`

---

### Example 8
**Source 1 Reference:**
- **Name**: `Laxmi Golden Investments Private Limited`
- **Address**: `New Bridge Business Centre'S 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, Karnataka`

**Matches in Source 2 / Source 3:**
- [Source 3] **Name**: `Laxmi Gbn lnvestments Private Limited`
- [Source 3] **Address**: `New Bridge Bssiness Centre's 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, ಕರ್ನಾಟಕ`
- [Source 3] **Name**: `Laxmi Golden Investments`
- [Source 3] **Address**: `New Bridge Buisness Centre's 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, KA`

---

### Example 9
**Source 1 Reference:**
- **Name**: `Hendricks and Flowers Inc`
- **Address**: `33 Sleepy Hollow Drive, Danbury, CT`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `Hendricks and  Flowers Inc`
- [Source 2] **Address**: `CT, SLEEPY HOLLOW DRIVE, DANBURY`
- [Source 3] **Name**: `Hendricks and Inc Flowers`
- [Source 3] **Address**: `nan`

---

### Example 10
**Source 1 Reference:**
- **Name**: `Hotel Enterprises Limited`
- **Address**: `Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi, West Delhi, Delhi`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `होटल एंटरप्राइजेज लिमिटेड`
- [Source 2] **Address**: `WZ-187C SHOP NO.13, DELHI, WEST DELHI, Delhi`
- [Source 3] **Name**: `Hotel Limited Services`
- [Source 3] **Address**: `Block B-517 Wz-187c Shop No.13, Divreportingcircle, West Delhi, DL`
- [Source 3] **Name**: `Hotel Énterprises Limited`
- [Source 3] **Address**: `Block B-517 Wz-187c Shop No.13, South West Delhi, Delhi, DL`

---

### Example 11
**Source 1 Reference:**
- **Name**: `Orellana Investments LLC`
- **Address**: `728 A Quail Avenue, Fl Ground Floor, Geneva, IA`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `Orellana Investments Investments Llc`
- [Source 2] **Address**: `nan`
- [Source 3] **Name**: `LLC Orellana Invsmbens`
- [Source 3] **Address**: `728 A Quail Avenue, Fl. Ground Floor, Geneva, Iowa`

---

### Example 12
**Source 1 Reference:**
- **Name**: `Chordia & Partners`
- **Address**: `Faridabad, 1038 Sector 9, Haryana`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `Chordia & Partners Company`
- [Source 2] **Address**: `हरियाणा, 1038 SECTOR 9, FARIDABAD`
- [Source 2] **Name**: `Chordia + Pagnters - 7306204978`
- [Source 2] **Address**: `हरियाणा, DOOR NO 1038 SECTOR 9, FARIDABAD`
- [Source 2] **Name**: `Chordia &-Pártners Ltd`
- [Source 2] **Address**: `H.NO 1038 SECTOR 9, FARIABAD, Haryana`
- [Source 3] **Name**: `Smt Chordia  & Center`
- [Source 3] **Address**: `#1038 Sector 9, Faridabad, हरियाणा`

---

### Example 13
**Source 1 Reference:**
- **Name**: `Crystal Staffing Solutions LLC`
- **Address**: `8706 Kentucky Derby Drive, Waxhaw, NC`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `Crystal Solutions LLC Partners`
- [Source 2] **Address**: `8706 KENTUCKY DERBY DR, WAXHAW, NC`
- [Source 2] **Name**: `CRYSTAL STAFFING SOLUTIONS-L.L.C.`
- [Source 2] **Address**: `8706 KENTUCKY DERBY DR, WAXHAW, NC`
- [Source 2] **Name**: `LLC Crystal Sttfrifng Solutions`
- [Source 2] **Address**: `8706 KENTUCKY DERBY DRIVE, WAXHAW, NC`
- [Source 3] **Name**: `Llc Crystal Staffing Solutions`
- [Source 3] **Address**: `870 Kentucky Derby Drive, Waxhaw, North Carolina`
- [Source 3] **Name**: `LLC Crystal Shaffing Solutions`
- [Source 3] **Address**: `870 Kentucky Derby Drive, Waxhaw, North Carolina`
- [Source 3] **Name**: `Crystal`
- [Source 3] **Address**: `8706 Kentucky Derby Drive, Waxhaw, NC`

---

### Example 14
**Source 1 Reference:**
- **Name**: `Obsidian, LLC`
- **Address**: `3907 Hamilton Road, Deer Park, WA`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `Obsidian,-LLC`
- [Source 2] **Address**: `3907 HAMILTON RD, DEER PARK CIYT, WA`
- [Source 2] **Name**: `obsidian, llc`
- [Source 2] **Address**: `nan`
- [Source 2] **Name**: `Obsidian, LLC Center`
- [Source 2] **Address**: `3907 HAMILTON RD, DEER PARK CIYT, WA`
- [Source 3] **Name**: `Obsidian, Llc`
- [Source 3] **Address**: `Deer Park, Washington, Hamilton Rd`
- [Source 3] **Name**: `Obsidian, [[LLC]]`
- [Source 3] **Address**: `nan`
- [Source 3] **Name**: `Korbrixx D.B.A. Obsidian, LLC`
- [Source 3] **Address**: `3907 Hamilton Road, Deer Park, WA`

---

### Example 15
**Source 1 Reference:**
- **Name**: `Dick Regional Armada Corp`
- **Address**: `33466 Warwick Hills Road, Yucaipa, CA`

**Matches in Source 2 / Source 3:**
- [Source 2] **Name**: `Dick Regional`
- [Source 2] **Address**: `nan`
- [Source 2] **Name**: `Dick Regional Armada`
- [Source 2] **Address**: `33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA`
- [Source 3] **Name**: `[Corp] Dick Regional Armada`
- [Source 3] **Address**: `Yucaipa, California, 33466 Warwik Hills Road`

---

