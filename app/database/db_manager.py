import aiosqlite
import os
import json
import datetime
from typing import List, Dict, Optional, Any, Union

DATABASE_PATH = os.environ.get("DATABASE_PATH", "data/financial_data.db")

async def init_db():
    # Create directory if it doesn't exist and if path contains a directory
    if os.path.dirname(DATABASE_PATH):
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Main client data table
        await db.execute('''
        CREATE TABLE IF NOT EXISTS client_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_tag TEXT NOT NULL,
            info TEXT NOT NULL,
            sensitivity_level INTEGER DEFAULT 1,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # User accounts table
        await db.execute('''
        CREATE TABLE IF NOT EXISTS user_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            account_number TEXT UNIQUE NOT NULL,
            balance REAL NOT NULL,
            account_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Transactions table
        await db.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            transaction_type TEXT NOT NULL,
            amount REAL NOT NULL,
            description TEXT,
            category TEXT,
            transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES user_accounts (id)
        )
        ''')
        
        await db.commit()

async def populate_sample_data():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Check if data already exists in client_data
        async with db.execute("SELECT COUNT(*) FROM client_data") as cursor:
            count = await cursor.fetchone()
            if count and count[0] > 0:
                return  # Data already exists
        
        # Sample client data with sensitivity levels
        # Higher sensitivity level means more sensitive information
        sample_data = [
            ('balance_info', 'Account balance: $5,432.10', 3),
            ('transaction_details', 'Last transaction: $200 debit for groceries at Whole Foods on Main St.', 2),
            ('credit_score', 'Your current credit score is 750 with Experian, 745 with TransUnion, and 752 with Equifax.', 3),
            ('investment_portfolio', 'Investment portfolio: $25,000 in tech stocks (AAPL, MSFT, GOOGL), $15,000 in bonds (Treasury, Municipal)', 3),
            ('loan_details', 'Mortgage: $250,000 at 3.5% interest rate, 25 years remaining. Account #1098-7654-3210', 3),
            ('account_security', 'Two-factor authentication is enabled on your account. Last login was from IP 192.168.1.1 on 2023-01-15', 2),
            ('transaction_history', 'Recent transactions: Restaurant $45.67 (01/10), Gas $30.25 (01/12), Online Shopping $125.99 (01/14)', 2),
            ('financial_advice', 'Based on your spending patterns, consider increasing your retirement contributions by 2% to maximize tax benefits', 1),
            ('market_update', 'Market is up 1.2% today, with technology sector leading gains of 2.5% and healthcare showing strong momentum', 1),
            ('tax_info', 'Your estimated tax return is $1,578.43 based on current filings. Your SSN ending in 1234 is associated with this return.', 3)
        ]
        
        await db.executemany(
            "INSERT INTO client_data (query_tag, info, sensitivity_level) VALUES (?, ?, ?)",
            sample_data
        )
        
        # Sample user accounts
        sample_accounts = [
            ('johndoe', '1234-5678-9012-3456', 5432.10, 'checking'),
            ('janedoe', '2468-1357-9080-7060', 3578.92, 'checking'),
            ('bobsmith', '9876-5432-1098-7654', 10250.75, 'savings'),
            ('sarahlee', '1357-2468-0909-8080', 25750.45, 'investment')
        ]
        
        await db.executemany(
            "INSERT INTO user_accounts (username, account_number, balance, account_type) VALUES (?, ?, ?, ?)",
            sample_accounts
        )
        
        # Get account IDs for transactions
        account_ids = {}
        async with db.execute("SELECT id, username, account_type FROM user_accounts") as cursor:
            async for row in cursor:
                key = f"{row[1]}_{row[2]}"
                account_ids[key] = row[0]
        
        # Sample transactions
        current_date = datetime.datetime.now()
        sample_transactions = []
        
        # Transactions for johndoe's checking account
        johndoe_checking_id = account_ids.get('johndoe_checking')
        if johndoe_checking_id:
            # Add transactions from the past 30 days
            for i in range(1, 31):
                date = current_date - datetime.timedelta(days=i)
                if i % 7 == 0:  # Weekly grocery shopping
                    sample_transactions.append((johndoe_checking_id, 'debit', 120.50, 'Grocery shopping at Whole Foods', 'groceries', date.strftime('%Y-%m-%d %H:%M:%S')))
                if i % 14 == 0:  # Bi-weekly dining out
                    sample_transactions.append((johndoe_checking_id, 'debit', 85.75, 'Dinner at Italian Restaurant', 'dining', date.strftime('%Y-%m-%d %H:%M:%S')))
                if i % 5 == 0:  # Regular coffee
                    sample_transactions.append((johndoe_checking_id, 'debit', 4.50, 'Coffee at Starbucks', 'coffee', date.strftime('%Y-%m-%d %H:%M:%S')))
            # Monthly salary
            sample_transactions.append((johndoe_checking_id, 'credit', 3500.00, 'Salary deposit from TechCorp Inc.', 'income', (current_date - datetime.timedelta(days=15)).strftime('%Y-%m-%d %H:%M:%S')))
        
        # Transactions for janedoe's checking account
        janedoe_checking_id = account_ids.get('janedoe_checking')
        if janedoe_checking_id:
            # Add transactions from the past 30 days
            for i in range(1, 31):
                date = current_date - datetime.timedelta(days=i)
                if i % 8 == 0:  # Weekly grocery shopping
                    sample_transactions.append((janedoe_checking_id, 'debit', 95.20, 'Grocery shopping at Trader Joe\'s', 'groceries', date.strftime('%Y-%m-%d %H:%M:%S')))
                if i % 10 == 0:  # Occasional dining out
                    sample_transactions.append((janedoe_checking_id, 'debit', 65.30, 'Lunch at Sushi Place', 'dining', date.strftime('%Y-%m-%d %H:%M:%S')))
            # Monthly salary
            sample_transactions.append((janedoe_checking_id, 'credit', 4200.00, 'Salary deposit from HealthCare Ltd.', 'income', (current_date - datetime.timedelta(days=12)).strftime('%Y-%m-%d %H:%M:%S')))
            
        # Transactions for bobsmith's savings account
        bobsmith_savings_id = account_ids.get('bobsmith_savings')
        if bobsmith_savings_id:
            # Monthly interest
            sample_transactions.append((bobsmith_savings_id, 'credit', 12.75, 'Interest payment', 'interest', (current_date - datetime.timedelta(days=5)).strftime('%Y-%m-%d %H:%M:%S')))
            # Occasional deposits
            sample_transactions.append((bobsmith_savings_id, 'credit', 500.00, 'Transfer from checking account', 'transfer', (current_date - datetime.timedelta(days=20)).strftime('%Y-%m-%d %H:%M:%S')))
            
        # Transactions for sarahlee's investment account
        sarahlee_investment_id = account_ids.get('sarahlee_investment')
        if sarahlee_investment_id:
            # Dividend payment
            sample_transactions.append((sarahlee_investment_id, 'credit', 320.45, 'Quarterly dividend payment', 'dividend', (current_date - datetime.timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S')))
            # Stock purchase
            sample_transactions.append((sarahlee_investment_id, 'debit', 1000.00, 'Purchase of AAPL shares', 'investment', (current_date - datetime.timedelta(days=25)).strftime('%Y-%m-%d %H:%M:%S')))
        
        if sample_transactions:
            await db.executemany(
                "INSERT INTO transactions (account_id, transaction_type, amount, description, category, transaction_date) VALUES (?, ?, ?, ?, ?, ?)",
                sample_transactions
            )
        
        await db.commit()

async def get_client_data(query_tag: str) -> List[Dict]:
    """Get client data based on query tag
    
    Args:
        query_tag: The type of data to retrieve
        
    Returns:
        List of matching data items
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, query_tag, info, sensitivity_level, last_updated FROM client_data WHERE query_tag = ?",
            (query_tag,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def add_client_data(query_tag: str, info: str) -> int:
    """Add new client data
    
    Args:
        query_tag: The type of data being added
        info: The information content
        
    Returns:
        ID of the newly inserted record
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO client_data (query_tag, info, sensitivity_level) VALUES (?, ?, 1)",
            (query_tag, info)
        )
        await db.commit()
        return cursor.lastrowid

async def get_account_balance(username: str, account_type: Optional[str] = None) -> List[Dict]:
    """Get account balance information for a user
    
    Args:
        username: The username to get balance for
        account_type: Optional filter for specific account types
        
    Returns:
        List of account information dictionaries
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        query = "SELECT id, account_number, balance, account_type FROM user_accounts WHERE username = ?"
        params = [username]
        
        if account_type:
            query += " AND account_type = ?"
            params.append(account_type)
            
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def get_recent_transactions(username: str, limit: int = 10) -> List[Dict]:
    """Get recent transactions for a user across all their accounts
    
    Args:
        username: The username to get transactions for
        limit: Maximum number of transactions to return
        
    Returns:
        List of transaction dictionaries
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        query = """
        SELECT t.id, t.transaction_type, t.amount, t.description, t.category, t.transaction_date, a.account_type, a.account_number
        FROM transactions t
        JOIN user_accounts a ON t.account_id = a.id
        WHERE a.username = ?
        ORDER BY t.transaction_date DESC
        LIMIT ?
        """
        
        async with db.execute(query, (username, limit)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def get_spending_analysis(username: str, days: int = 30) -> Dict[str, Any]:
    """Get spending analysis by category for a user
    
    Args:
        username: The username to analyze
        days: Number of days to analyze
        
    Returns:
        Dictionary with spending analysis data
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        # Calculate the date threshold
        date_threshold = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Get total spending
        total_query = """
        SELECT SUM(t.amount) as total_spent
        FROM transactions t
        JOIN user_accounts a ON t.account_id = a.id
        WHERE a.username = ? AND t.transaction_type = 'debit' AND t.transaction_date >= ?
        """
        
        async with db.execute(total_query, (username, date_threshold)) as cursor:
            total_row = await cursor.fetchone()
            total_spent = total_row['total_spent'] if total_row and total_row['total_spent'] else 0
        
        # Get spending by category
        category_query = """
        SELECT t.category, SUM(t.amount) as amount
        FROM transactions t
        JOIN user_accounts a ON t.account_id = a.id
        WHERE a.username = ? AND t.transaction_type = 'debit' AND t.transaction_date >= ?
        GROUP BY t.category
        ORDER BY amount DESC
        """
        
        categories = []
        async with db.execute(category_query, (username, date_threshold)) as cursor:
            async for row in cursor:
                categories.append({
                    'category': row['category'],
                    'amount': row['amount'],
                    'percentage': (row['amount'] / total_spent * 100) if total_spent > 0 else 0
                })
        
        return {
            'total_spent': total_spent,
            'period_days': days,
            'categories': categories,
            'analysis_date': datetime.datetime.now().strftime('%Y-%m-%d')
        }

async def get_all_recent_transactions(limit: int = 10) -> List[Dict]:
    """Get recent transactions across all accounts in the system
    
    Args:
        limit: Maximum number of transactions to return
        
    Returns:
        List of transaction dictionaries
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        query = """
        SELECT t.id, t.transaction_type, t.amount, t.description, t.category, t.transaction_date, 
               a.account_type, a.account_number, a.username
        FROM transactions t
        JOIN user_accounts a ON t.account_id = a.id
        ORDER BY t.transaction_date DESC
        LIMIT ?
        """
        
        async with db.execute(query, (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
