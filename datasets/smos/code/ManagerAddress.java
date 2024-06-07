package smos.storage;
import smos.bean.Address;
import smos.bean.Teaching;
import smos.exception.DuplicatedEntityException;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.exception.MandatoryFieldException;
import smos.storage.connectionManagement.DBConnection;
import smos.storage.connectionManagement.exception.ConnectionException;
import smos.utility.Utility;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Vector;
import java.sql.Connection;

/**
*
* Address Manager Class
*
*/

public class ManagerAddress {

	private static ManagerAddress instance;
	
     /**
	 * The name of the address table
	 */
	public static final String TABLE_ADDRESS = "address";
	public static final String TABLE_ADDRESS_HAS_TEACHING = "address_has_teaching";
	
	private ManagerAddress() {
		super();
	}
	

	/**
	 * Returns the only instance of the existing class.
	 *
	 * @return Returns the instance of the class.
	 */
	public static synchronized ManagerAddress getInstance(){
		if(instance==null){
			instance = new ManagerAddress();
			}
			return instance;
		}
	/**
	 * Verify the existence of an address in the database.
	 * 
	 * @param pAddress
	 *            Address to check.
	 * @return true if the address passed as a parameter already exists,
	 * 			otherwise false.
	 *  
	 * @throws MandatoryFieldException 
	 * @throws SQLException
	 * @throws MandatoryFieldException
	 * @throws ConnectionException 
	 * @throws ConnectionException
	 * @throws SQLException 
	 */
	
	public synchronized boolean hasTeaching(Teaching pTeaching, Address pAddress)
	throws SQLException, EntityNotFoundException, ConnectionException,
	InvalidValueException {
Connection connect = null;
boolean result = false;
if (pTeaching.getId() <= 0)
	throw new EntityNotFoundException("Specify teaching");

try {
	connect = DBConnection.getConnection();
	if (connect == null)
		throw new ConnectionException();

	// Prepare sql string
	String sql = "SELECT * FROM "
			+ ManagerTeaching.TABLE_ADDRESS_TEACHINGS
			+ " WHERE id_teaching = "
			+ Utility.isNull(pTeaching.getId())
			+" AND id_address = "
			+ Utility.isNull(pAddress.getIdAddress());
	// Send the Query to the database
	ResultSet pRs = Utility.queryOperation(connect, sql);
	if (pRs.next())
		result = true;

	return result;

} finally {
	// we release resources
	DBConnection.releaseConnection(connect);

}
}

	public synchronized boolean exists (Address pAddress) throws MandatoryFieldException, ConnectionException, SQLException {
	boolean result= false;
	Connection connect = null;
	
	if (pAddress.getName() == null)
		throw new MandatoryFieldException("Specificare il nome.");
	try{
		//We get the connection to the database.
		connect= DBConnection.getConnection();
		
		if (connect == null)
			throw new ConnectionException();
		
		String sql =" SELECT * FROM "
		+ ManagerAddress.TABLE_ADDRESS 
		+ " WHERE name = "
		+ Utility.isNull(pAddress.getName());
		
		ResultSet tRs = Utility.queryOperation(connect, sql);
		
		if(tRs.next())
			result = true;
		
		return result;
		
	}
	finally{
		DBConnection.releaseConnection(connect);
	}
	}
	/**
	 * Inserts a new address in the address table.
	 * 
	 * @param pAddress 
	 * 			The address to enter.
	 * 
	 * @throws SQLException
	 * @throws ConnectionException
	 * @throws MandatoryFieldException 
	 * @throws EntityNotFoundException  
	 * @throws InvalidValueException 
	 */
	
	public synchronized void insert(Address pAddress) 
	throws MandatoryFieldException, ConnectionException, 
	SQLException, EntityNotFoundException, 
	InvalidValueException{
	Connection connect= null;
try{
// control of mandatory fields
if(pAddress.getName()==null)
	throw new MandatoryFieldException("Specify the name field");

	connect = DBConnection.getConnection();
if (connect==null)
	throw new ConnectionException();
	//Prepare Sql string
	String sql =
	"INSERT INTO " 
	+ ManagerAddress.TABLE_ADDRESS 
	+ " (name) " 
	+ "VALUES (" 
	+ Utility.isNull(pAddress.getName()) 
	+ ")";

	Utility.executeOperation(connect,sql);

	pAddress.setIdAddress(Utility.getMaxValue("id_address",ManagerAddress.TABLE_ADDRESS));

	}finally {
		//releases resources

		DBConnection.releaseConnection(connect);
}
}
	/**
	 * Delete an address from the address table.
	 * 
	 * @param pAddress 
	 * 			The address to delete.
	 * 
	 * @throws MandatoryFieldException 
	 * @throws EntityNotFoundException 
	 * @throws SQLException 
	 * @throws ConnectionException 
	 * @throws InvalidValueException 
	 * 
	 */
	public synchronized void delete (Address pAddress) throws ConnectionException, 
			SQLException, EntityNotFoundException, MandatoryFieldException, InvalidValueException {
		Connection connect = null;

		
		try {
			//ManagerAddress.getInstance().AddressOnDeleteCascade(pAddress);
			connect = DBConnection.getConnection();
				//Prepare SQL string
				String sql = "DELETE FROM " 
							+ ManagerAddress.TABLE_ADDRESS 
							+ " WHERE id_address = "
							+ Utility.isNull(pAddress.getIdAddress());
			
				Utility.executeOperation(connect, sql);
		}  
		finally {
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}
	public synchronized void assignTeachingAsAddress (Address pAddress, Teaching pTeaching) throws ConnectionException, 
	SQLException, EntityNotFoundException, MandatoryFieldException, InvalidValueException, DuplicatedEntityException {
Connection connect = null;
ManagerAddress managerAddress = ManagerAddress.getInstance();
if(managerAddress.hasTeaching(pTeaching, pAddress))
	throw new DuplicatedEntityException("This address already has this associated teaching");

try {
	//ManagerAddress.getInstance().AddressOnDeleteCascade(pAddress);
	connect = DBConnection.getConnection();
		//Prepare SQL string
		String sql = "INSERT INTO " 
					+ ManagerAddress.TABLE_ADDRESS_HAS_TEACHING
					+ " (id_address, id_teaching) "
					+ " VALUES( "
					+ Utility.isNull(pAddress.getIdAddress())
					+ " , "
					+ Utility.isNull(pTeaching.getId())
					+ " )";
	
		Utility.executeOperation(connect, sql);
}  
finally {
	//releases resources
	DBConnection.releaseConnection(connect);
}
}
	public synchronized void removeTeachingAsAddress (Address pAddress, Teaching pTeaching) throws ConnectionException, 
	SQLException, EntityNotFoundException, MandatoryFieldException, InvalidValueException {
Connection connect = null;
ManagerAddress managerAddress = ManagerAddress.getInstance();
if(!managerAddress.hasTeaching(pTeaching, pAddress))
	throw new EntityNotFoundException("This address does not contain the teaching to remove");

try {
	//ManagerAddress.getInstance().AddressOnDeleteCascade(pAddress);
	connect = DBConnection.getConnection();
		//Prepare SQL string
		String sql = "DELETE FROM " 
					+ ManagerAddress.TABLE_ADDRESS_HAS_TEACHING
					+ " WHERE id_address= "
					+ Utility.isNull(pAddress.getIdAddress())		
					+ " AND id_teaching = "
					+ Utility.isNull(pTeaching.getId());
	
		Utility.executeOperation(connect, sql);
}  
finally {
	//releases resources
	DBConnection.releaseConnection(connect);
}
}

	/**
	 * Returns the id of the address passed as a parameter.
	 * 
	 * @param pAddress
	 *            The address whose id is requested.
	 * @return the id of the address passed as a parameter.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	public synchronized int getAddressId(Address pAddress)
			throws EntityNotFoundException, ConnectionException, SQLException {
		int result = 0;
		Connection connect = null;
		try {
			if (pAddress == null)
				throw new EntityNotFoundException("Can't find the address!");

			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the id of the address passed as a parameter.
			 */
			String tSql = "SELECT id_address FROM " 
				+ ManagerAddress.TABLE_ADDRESS
				+ " WHERE name = " 
				+ Utility.isNull(pAddress.getName());

			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);

			if (tRs.next())
				result = tRs.getInt("id_address");

			return result;
		} finally {
			DBConnection.releaseConnection(connect);
		}
	}
	/**
	 * Returns the address corresponding to the id passed as a parameter.
	 * 
	 * @param pIdAddress
	 * 			The address id.
	 * @return the address associated with the past id as a parameter.
	 * 
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException 
	 */
	public synchronized Address getAddressById (int pIdAddress) throws ConnectionException, SQLException, EntityNotFoundException, InvalidValueException{
		Address result = null;
		Connection connect = null;
		try
		{
						
			if (pIdAddress <= 0) 
				throw new EntityNotFoundException("Could not find address!");
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			// Prepare SQL string
			String sql = 
				"SELECT * FROM " 
				+ ManagerAddress.TABLE_ADDRESS
				+ " WHERE id_address = " 
				+ Utility.isNull(pIdAddress);
			
			// Send Query to DataBase
			ResultSet pRs = Utility.queryOperation(connect, sql);
			
			if (pRs.next()) 
				result = this.loadRecordFromRs(pRs);
			else 
				throw new EntityNotFoundException("Unable to find user!");
							
			return result;
		}finally{
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}
	/**
	 * Returns the set of all addresses in the database.
	 * 
	 * @return a collection of addresses.
	 * 
	 * @throws ConnectionException 
	 * @throws EntityNotFoundException 
	 * @throws SQLException 
	 * @throws InvalidValueException 
	 */
	public synchronized Collection<Address> getAddressList() throws ConnectionException, EntityNotFoundException, SQLException, InvalidValueException{
		Connection connect = null;
		Collection<Address> result = new Vector<Address>();;
		
		try {
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			//Prepare sql string
			String sql = "SELECT * "  
				+ " FROM " 
				+ ManagerAddress.TABLE_ADDRESS 
				+ " ORDER BY id_address";
				
			//Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, sql);
			
			if (tRs.next())
				result = this.loadRecordsFromRs(tRs);		
			return result;
		} finally {
			//releases resources		
			DBConnection.releaseConnection(connect);
		}
	}
	/**
	 * Return the name of the address corresponding to the id
	 * passed as parameter.
	 * 
	 * @param pIdAddress
	 * 			The address id.
	 * @return Returns a string containing the address name.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	public synchronized String getAddressNameById(int pIdAddress) throws EntityNotFoundException, ConnectionException, SQLException{
		String result;
		Connection connect = null;
		try
		{
			// If the id has not been provided we return an error code
			if (pIdAddress <= 0)
				throw new EntityNotFoundException("Could not find address!");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the user id passed as parameter
			 */
			String tSql = 
				"SELECT name FROM " 
				+ ManagerAddress.TABLE_ADDRESS 
				+ " WHERE id_address = " 
				+ Utility.isNull(pIdAddress) ;
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			if (tRs.next()) 
				result = tRs.getString("name");
			else 
				throw new EntityNotFoundException("Could not find address!");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	
	/**
	 * Return a collection with the id of the teachings associated with the id
	 * passed as parameter.
	 * 
	 * @param pIdAddress
	 * 			The address id.
	 * @return a collection with int
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	
	public synchronized Collection<Integer> getAddressTechings(int pIdAddress) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException{
		Collection<Integer> result;
		Connection connect = null;
		try
		{
			// If the id has not been provided we return an error code
			if (pIdAddress <= 0)
				throw new EntityNotFoundException("Could not find address!");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the user id passed as parameter
			 */
			String tSql = 
				"SELECT id_teaching FROM " 
				+ ManagerAddress.TABLE_ADDRESS_HAS_TEACHING
				+ " WHERE id_address = " 
				+ Utility.isNull(pIdAddress) ;
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			result = this.loadIntegersFromRs(tRs);
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}

	/*
	 * Lets you read a record from the ResultSet.
	 */
	private Address loadRecordFromRs(ResultSet pRs) throws SQLException, InvalidValueException{
		Address address = new Address();
		address.setName(pRs.getString("name"));
		address.setIdAddress(pRs.getInt("id_address"));
		return address;
	}
	
	/*
	 * Allows you to read records from the ResultSet.
	 */
	private Collection<Address> loadRecordsFromRs(ResultSet pRs) throws SQLException, InvalidValueException{
		Collection<Address> result = new Vector<Address>();
		do  {
			result.add(loadRecordFromRs(pRs));
		} while (pRs.next());
		return result;
	}
	
	private Collection<Integer> loadIntegersFromRs(ResultSet pRs) throws SQLException, InvalidValueException{
		Collection<Integer> result = new Vector<Integer>();
		while(pRs.next())  {
			result.add(pRs.getInt("id_teaching"));
		} 
		return result;
	}
	
		
	
}
