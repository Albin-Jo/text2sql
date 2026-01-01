-- Finance Domain Schema for Text-to-SQL MVP
-- BigQuery compatible DDL

-- Customers table
CREATE TABLE customers (
  customer_id STRING NOT NULL,
  first_name STRING NOT NULL,
  last_name STRING NOT NULL,
  email STRING,
  phone STRING,
  date_of_birth DATE,
  created_at TIMESTAMP NOT NULL,
  status STRING DEFAULT 'active',
  credit_score INT64,
  PRIMARY KEY (customer_id)
);

-- Branches table
CREATE TABLE branches (
  branch_id STRING NOT NULL,
  branch_name STRING NOT NULL,
  city STRING NOT NULL,
  state STRING,
  country STRING DEFAULT 'USA',
  opened_date DATE,
  manager_id STRING,
  PRIMARY KEY (branch_id)
);

-- Employees table
CREATE TABLE employees (
  employee_id STRING NOT NULL,
  first_name STRING NOT NULL,
  last_name STRING NOT NULL,
  email STRING,
  hire_date DATE NOT NULL,
  position STRING NOT NULL,
  salary NUMERIC(12, 2),
  branch_id STRING NOT NULL,
  manager_employee_id STRING,
  PRIMARY KEY (employee_id),
  FOREIGN KEY (branch_id) REFERENCES branches(branch_id)
);

-- Accounts table
CREATE TABLE accounts (
  account_id STRING NOT NULL,
  customer_id STRING NOT NULL,
  account_type STRING NOT NULL,
  balance NUMERIC(15, 2) DEFAULT 0.00,
  currency STRING DEFAULT 'USD',
  opened_date DATE NOT NULL,
  closed_date DATE,
  status STRING DEFAULT 'active',
  interest_rate NUMERIC(5, 4),
  branch_id STRING,
  PRIMARY KEY (account_id),
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
  FOREIGN KEY (branch_id) REFERENCES branches(branch_id)
);

-- Transactions table
CREATE TABLE transactions (
  transaction_id STRING NOT NULL,
  account_id STRING NOT NULL,
  transaction_type STRING NOT NULL,
  amount NUMERIC(15, 2) NOT NULL,
  transaction_date TIMESTAMP NOT NULL,
  description STRING,
  category STRING,
  merchant STRING,
  status STRING DEFAULT 'completed',
  reference_number STRING,
  PRIMARY KEY (transaction_id),
  FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- Loans table
CREATE TABLE loans (
  loan_id STRING NOT NULL,
  customer_id STRING NOT NULL,
  loan_type STRING NOT NULL,
  principal_amount NUMERIC(15, 2) NOT NULL,
  interest_rate NUMERIC(5, 4) NOT NULL,
  term_months INT64 NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE,
  monthly_payment NUMERIC(12, 2),
  remaining_balance NUMERIC(15, 2),
  status STRING DEFAULT 'active',
  branch_id STRING,
  PRIMARY KEY (loan_id),
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
  FOREIGN KEY (branch_id) REFERENCES branches(branch_id)
);
