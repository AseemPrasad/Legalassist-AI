
import os
import structlog
from fpdf import FPDF
import json
import sys

def create_pdf(path, content):
    """Create a PDF with the given content"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, content)
    pdf.output(path)

def main(num_fixtures=50):
    """Generate test fixtures for all case types"""
    base_dir = "tests/samples"
    
    # Create directories for all case types
    case_types = {
        "criminal": ["guilty", "acquitted"],
        "civil": ["plaintiff_won", "plaintiff_lost"],
        "family": ["custody_granted", "custody_denied"],
        "labor": ["termination_upheld", "termination_overturned"],
        "landlord_tenant": ["eviction_granted", "eviction_denied"],
    }
    
    for case_type, subtypes in case_types.items():
        for subtype in subtypes:
            os.makedirs(f"{base_dir}/{case_type}/{subtype}", exist_ok=True)
    
    test_data = []
    case_counter = 0
    per_type = max(1, num_fixtures // len(case_types))
    
    # Criminal Guilty (10 cases)
    for i in range(1, min(per_type // 2 + 1, 11)):
        content = f"""JUDGMENT
        
IN THE COURT OF SESSIONS, NEW DELHI
CASE NO: CR/IND/{100+i}/2023
STATE VS JOHN DOE {i}

Facts: The accused is charged under Section 379 of IPC for theft of property worth rupees five thousand.
The accused was found with stolen goods in possession. Fingerprints match.

Evidence: Three witnesses testified that accused was at the crime scene. Forensic report confirms fingerprints.
CCTV footage corroborates the narrative. Recovered stolen items from accused's residence.

Finding: The prosecution has proved the case beyond reasonable doubt.

JUDGMENT: The accused is found GUILTY under Section 379 IPC.

Sentencing: Imprisonment for 2 years and fine of rupees 5000.

Appeal: Defendant has the right to appeal within 90 days in the High Court."""
        path = f"{base_dir}/criminal/guilty/case_{i}.pdf"
        create_pdf(path, content)
        test_data.append({
            "path": path,
            "type": "criminal_guilty",
            "expected_verdict": "guilty",
            "expected_appeal": "yes",
            "expected_days": "90",
            "case_value": "5000"
        })
        case_counter += 1
    
    # Criminal Acquitted (10 cases)
    for i in range(1, min(per_type // 2 + 1, 11)):
        content = f"""JUDGMENT
        
IN THE COURT OF SESSIONS, MUMBAI
CASE NO: CR/MUM/{200+i}/2023
STATE VS JANE SMITH {i}

Facts: Accused is charged under Section 323 IPC for voluntarily causing hurt to the complainant.

Evidence: Prosecution relies on medical certificate showing minor injuries. Complainant is not present.
Key witness turned hostile during cross-examination. No corroborating evidence.

Finding: The testimony is fraught with doubt and contradictions. Prosecution failed to discharge burden.

JUDGMENT: The accused is ACQUITTED of all charges due to lack of credible evidence.

The accused is free to go. No compensation ordered."""
        path = f"{base_dir}/criminal/acquitted/case_{i}.pdf"
        create_pdf(path, content)
        test_data.append({
            "path": path,
            "type": "criminal_acquitted",
            "expected_verdict": "acquitted",
            "expected_appeal": "rarely",
            "expected_days": "90",
        })
        case_counter += 1
    
    # Civil Plaintiff Won (10 cases)
    for i in range(1, min(per_type // 2 + 1, 11)):
        content = f"""JUDGMENT
        
IN THE CIVIL COURT, BANGALORE
CASE NO: CS/BNG/{300+i}/2023
ALICE {i} VS BOB {i}

Subject Matter: Suit for recovery of possession of immovable property and damages.

Facts: Plaintiff claims ownership by virtue of registered sale deed dated 2020.
Defendant claims adverse possession but lacks documentary evidence.

Evidence: Plaintiff produced: (1) Sale deed, (2) Property tax receipts, (3) Possession since 2020.
Defendant produced no credible evidence and failed to appear on several dates.

Finding: On preponderance of probabilities, plaintiff has proved her case.

JUDGMENT: THIS SUIT IS DECREED in favor of the plaintiff. 
Possession of property shall be handed over to plaintiff within 60 days.
Defendant shall pay costs rupees 10,000."""
        path = f"{base_dir}/civil/plaintiff_won/case_{i}.pdf"
        create_pdf(path, content)
        test_data.append({
            "path": path,
            "type": "civil_won",
            "expected_verdict": "decreed",
            "expected_appeal": "yes",
            "expected_days": "30",
            "case_value": "500000"
        })
        case_counter += 1
    
    # Civil Plaintiff Lost (10 cases)
    for i in range(1, min(per_type // 2 + 1, 11)):
        content = f"""JUDGMENT
        
IN THE CIVIL COURT, CHENNAI
CASE NO: CS/CHE/{400+i}/2023
CHARLIE {i} VS DAVID {i}

Subject Matter: Suit for recovery of money advanced as loan.

Facts: Plaintiff claims to have advanced rupees 5 lakhs to defendant as a friendly loan.
No written agreement, promissory note, or witness to the transaction.

Evidence: Plaintiff's own testimony is inconsistent. Defendant denies borrowing any money.
No documentary proof of transaction. ATM statements not produced.

Finding: Testimony is weak and not corroborated. Suit is also barred by limitation as alleged loan was in 2015.

JUDGMENT: THIS SUIT IS DISMISSED. Plaintiff fails to prove the claim.
No relief granted. Defendant is absolved of liability."""
        path = f"{base_dir}/civil/plaintiff_lost/case_{i}.pdf"
        create_pdf(path, content)
        test_data.append({
            "path": path,
            "type": "civil_lost",
            "expected_verdict": "dismissed",
            "expected_appeal": "yes",
            "expected_days": "30",
            "case_value": "500000"
        })
        case_counter += 1
    
    # Family Law - Custody Granted (6 cases)
    for i in range(1, 7):
        content = f"""JUDGMENT
        
IN THE FAMILY COURT, DELHI
CASE NO: FA/{500+i}/2023
MOTHER {i} VS FATHER {i}

Subject Matter: Custody of minor child aged 8 years.

Facts: Mother filed application for custody after separation. Child was with father for past 2 years.
Father claims better financial situation. Mother claims stronger emotional bond.

Evidence: Psychologist report states child is emotionally attached to mother.
School teacher confirms child mentions mother frequently. Both parents are capable.

Finding: Welfare of child requires continuous nurturing, which mother is better positioned to provide.

JUDGMENT: CUSTODY OF THE CHILD IS GRANTED to the mother.
Father has visitation rights on weekends and holidays.
Maintenance of rupees 5000 per month to be paid by father."""
        path = f"{base_dir}/family/custody_granted/case_{i}.pdf"
        create_pdf(path, content)
        test_data.append({
            "path": path,
            "type": "family_custody_granted",
            "expected_verdict": "custody_granted",
            "expected_appeal": "yes",
            "expected_days": "30",
        })
        case_counter += 1
    
    # Family Law - Custody Denied (4 cases)
    for i in range(1, 5):
        content = f"""JUDGMENT
        
IN THE FAMILY COURT, MUMBAI
CASE NO: FA/{600+i}/2023
FATHER {i} VS MOTHER {i}

Subject Matter: Custody modification of minor child.

Facts: Father seeks modification of earlier custody order from mother to himself.
Claims mother has remarried and neglecting child. Child is 10 years old.

Evidence: Child welfare officer report finds mother is caring for child properly.
School records show good attendance and performance. No credible evidence of neglect.

Finding: Modification would not serve the interests of the child. Current arrangement is satisfactory.

JUDGMENT: APPLICATION FOR MODIFICATION OF CUSTODY IS REJECTED.
Child shall remain with mother. Existing arrangement continues."""
        path = f"{base_dir}/family/custody_denied/case_{i}.pdf"
        create_pdf(path, content)
        test_data.append({
            "path": path,
            "type": "family_custody_denied",
            "expected_verdict": "custody_denied",
            "expected_appeal": "yes",
            "expected_days": "30",
        })
        case_counter += 1
    
    # Labor - Termination Upheld (4 cases)
    for i in range(1, 5):
        content = f"""JUDGMENT
        
IN THE LABOUR COURT, BANGALORE
CASE NO: LB/{700+i}/2023
EMPLOYEE {i} VS COMPANY {i}

Subject Matter: Termination of employment and claim for compensation.

Facts: Employee was terminated on grounds of misconduct - repeated absenteeism and insubordination.
Company claims employee absented 45 days in 6 months without leave. Employee claims discrimination.

Evidence: Attendance records show 45 absences. Warnings issued on 3 occasions. Employee failed to respond to notices.

Finding: Employer has followed due process. Grounds for termination are justified and established.

JUDGMENT: TERMINATION OF EMPLOYMENT IS UPHELD.
No compensation payable to employee. Claim dismissed."""
        path = f"{base_dir}/labor/termination_upheld/case_{i}.pdf"
        create_pdf(path, content)
        test_data.append({
            "path": path,
            "type": "labor_termination_upheld",
            "expected_verdict": "upheld",
            "expected_appeal": "yes",
            "expected_days": "90",
        })
        case_counter += 1
    
    # Labor - Termination Overturned (2 cases)
    for i in range(1, 3):
        content = f"""JUDGMENT
        
IN THE LABOUR COURT, HYDERABAD
CASE NO: LB/{800+i}/2023
EMPLOYEE {i} VS EMPLOYER {i}

Subject Matter: Wrongful termination and reinstatement claim.

Facts: Employee of 15 years was terminated without notice or hearing alleging poor performance.
No written warning was issued. Employee claims victimization for union activities.

Evidence: No evidence of poor performance. Service record shows satisfactory evaluations.
Witness testimonies corroborate union membership. Termination appears unjustified.

Finding: Termination was not based on proper grounds. Procedure not followed. Violation of labor laws.

JUDGMENT: TERMINATION IS SET ASIDE. Employee shall be reinstated with back wages.
Back wages for 12 months shall be paid with interest at 6% per annum."""
        path = f"{base_dir}/labor/termination_overturned/case_{i}.pdf"
        create_pdf(path, content)
        test_data.append({
            "path": path,
            "type": "labor_termination_overturned",
            "expected_verdict": "overturned",
            "expected_appeal": "yes",
            "expected_days": "90",
        })
        case_counter += 1
    
    # Landlord-Tenant - Eviction Granted (4 cases)
    for i in range(1, 5):
        content = f"""JUDGMENT
        
IN THE PROPERTY COURT, KOLKATA
CASE NO: PR/{900+i}/2023
LANDLORD {i} VS TENANT {i}

Subject Matter: Eviction suit on grounds of non-payment of rent.

Facts: Tenant has not paid rent for past 8 months. Landlord issued 3 notices. Rent agreed: rupees 10,000 per month.
Arrears: rupees 80,000. Tenant claims financial hardship but offers no alternative arrangement.

Evidence: Rent receipts showing payment up to 4 months ago. Three demand notices dated and served properly.

Finding: Non-payment of rent is established. Grounds for eviction are satisfied.

JUDGMENT: EVICTION ORDER IS GRANTED.
Tenant shall vacate the property within 90 days. Arrear rent plus damages to be recovered.
Landlord free to take possession after 90 days."""
        path = f"{base_dir}/landlord_tenant/eviction_granted/case_{i}.pdf"
        create_pdf(path, content)
        test_data.append({
            "path": path,
            "type": "landlord_tenant_eviction_granted",
            "expected_verdict": "eviction_granted",
            "expected_appeal": "yes",
            "expected_days": "30",
        })
        case_counter += 1
    
    # Landlord-Tenant - Eviction Denied (2 cases)
    for i in range(1, 3):
        content = f"""JUDGMENT
        
IN THE PROPERTY COURT, PUNE
CASE NO: PR/{1000+i}/2023
LANDLORD {i} VS TENANT {i}

Subject Matter: Eviction suit on grounds of unauthorized occupation.

Facts: Landlord claims tenant has permitted unauthorized persons to stay in property.
Tenant claims family members who are authorized occupants. One person is son, another is daughter.

Evidence: Rent paid regularly. Family members have been mentioned in previous communications.
No evidence of commercial subletting. Reasonable family occupation.

Finding: Use of property by family members does not constitute unauthorized occupation.

JUDGMENT: EVICTION SUIT IS DISMISSED.
Tenant's occupation is lawful. Landlord's suit rejected. Tenant to continue possession."""
        path = f"{base_dir}/landlord_tenant/eviction_denied/case_{i}.pdf"
        create_pdf(path, content)
        test_data.append({
            "path": path,
            "type": "landlord_tenant_eviction_denied",
            "expected_verdict": "eviction_denied",
            "expected_appeal": "yes",
            "expected_days": "30",
        })
        case_counter += 1
    
    # Save metadata
    with open("tests/test_metadata.json", "w") as f:
        json.dump(test_data, f, indent=4)
    
    print(f"✓ Generated {case_counter} test fixtures ({len(case_types)} case types)")
    print(f"✓ Created test_metadata.json with {len(test_data)} entries")
    print(f"✓ Test cases cover: {', '.join(case_types.keys())}")
    return case_counter

if __name__ == "__main__":
    num_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    main(num_cases)
