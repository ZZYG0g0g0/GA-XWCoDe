package smos.storage;


import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Iterator;
import java.util.Vector;

import smos.bean.Classroom;
import smos.bean.User;

import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.exception.MandatoryFieldException;
import smos.storage.connectionManagement.DBConnection;
import smos.storage.connectionManagement.exception.ConnectionException;
import smos.utility.Utility;

public class ManagerClassroom  {

	/**
	 * Class that manages the classes of the institution
	 * @author Nicola Pisanti
	 * @version 1.0
	 */
	
	private static ManagerClassroom instance;
	
	public static final String TABLE_CLASSROOM = "classroom";
	public static final String TABLE_ADDRESS ="address";
	public static final String TABLE_TEACHER_HAS_CLASSROOM = "teacher_has_classroom";
	public static final String TABLE_STUDENT_HAS_CLASSROOM = "student_has_classroom";
	
	private ManagerClassroom(){
		super();
	}
	
	
	
	/**
	 * Returns the only instance of the existing class.
	 * 
	 * @return the class instance.
	 */
	public static synchronized ManagerClassroom getInstance(){
		if(instance==null){
			instance = new ManagerClassroom();
		}
		return instance;
	}
	
	
	
	
	
	/**
	 * Check if the input given class is in the database
	 * @param The class whose existence must be verified
	 * @return true if the class is in the database, otherwise false
	 * @throws MandatoryFieldException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	public synchronized boolean exists (Classroom pClassroom) throws MandatoryFieldException, ConnectionException, SQLException {
		
		boolean result = false;
		Connection connect = null;

		if (pClassroom.getName() == null)
			throw new MandatoryFieldException("Specify the class name.");
		if (pClassroom.getAcademicYear() <=1970)
			throw new MandatoryFieldException("Specify the academic year");
		if (pClassroom.getIdAddress()<=0){
			throw new MandatoryFieldException("Specify address");
			//the user enters the address, is converted to idAddress
		}
		
		try {
			//We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			String sql = "SELECT * FROM " 
				+ ManagerClassroom.TABLE_CLASSROOM
				+ " WHERE name = " 
				+ Utility.isNull(pClassroom.getName()) 
				+ " AND accademic_year = "
				+ Utility.isNull(pClassroom.getAcademicYear()
				+ " AND id_address = "
				+ Utility.isNull(pClassroom.getIdAddress())
				
				);

			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, sql);

			if (tRs.next())
				result = true;

			return result;
			
		} finally {
			DBConnection.releaseConnection(connect);
		}
	}
	
	
	
	/**
	 * Inserts the class object in the database
     * @param the class to enter in the database
	 * @throws MandatoryFieldException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void insert(Classroom pClassroom) throws MandatoryFieldException, 
		ConnectionException, SQLException, EntityNotFoundException, InvalidValueException{
		
		Connection connect= null;
		try{
			// control of mandatory fields
			if (pClassroom.getName() == null)
				throw new MandatoryFieldException("Specify the class name.");
			if (pClassroom.getAcademicYear() <=1970)
				throw new MandatoryFieldException("Specify the academic year");
			if (pClassroom.getIdAddress()<=0){
				throw new MandatoryFieldException("Specify address");
				//the user enters the address, is converted to idAddress
			}
			
			connect = DBConnection.getConnection();
			if (connect==null)
				throw new ConnectionException();
			//Prepare Sql string
			String sql =
				"INSERT INTO " 
				+ ManagerClassroom.TABLE_CLASSROOM 
				+ " (id_address, name, accademic_year) " 
				+ "VALUES (" 
				+ Utility.isNull(pClassroom.getIdAddress()) 
				+ "," 
				+ Utility.isNull(pClassroom.getName()) 
				+ "," 
				+ Utility.isNull(pClassroom.getAcademicYear())
				+ ")";
		
			Utility.executeOperation(connect,sql);
		
			pClassroom.setIdClassroom((Utility.getMaxValue("id_classroom",ManagerClassroom.TABLE_CLASSROOM)));
		
		}finally {
		//releases resources
		
		DBConnection.releaseConnection(connect);
		}
	}
	
	/**
	 * Update the statistics of a class
     * @param Class with updated statistics (but identical ID)
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws MandatoryFieldException
	 */
	public synchronized void update (Classroom pClassroom) throws ConnectionException,
	SQLException, EntityNotFoundException, MandatoryFieldException{
		Connection connect= null;
		
		try{
			if (pClassroom.getIdClassroom()<=0)
				throw new EntityNotFoundException("Couldn't find the class!");
			
			if (pClassroom.getName() == null)
				throw new MandatoryFieldException("Specify the class name.");
			if (pClassroom.getAcademicYear() <=1970)
				throw new MandatoryFieldException("Specify the academic year");
			if (pClassroom.getIdAddress()<=0){
				throw new MandatoryFieldException("Specify address");
				//the user enters the address, is converted to idAddress
			}
			//Prepare SQL string
			String sql=
				"UPDATE " 
				+	ManagerClassroom.TABLE_CLASSROOM 
				+ " SET" 
				+ " id_address = " 
				+ Utility.isNull(pClassroom.getIdAddress()) 
				+ ", name = " 
				+ Utility.isNull(pClassroom.getName()) 
				+ ", accademic_year = " 
				+ Utility.isNull(pClassroom.getAcademicYear())  
				+ " WHERE id_classroom = " 
				+ Utility.isNull(pClassroom.getIdClassroom());
			
			//make a new connection and send the query
			connect = DBConnection.getConnection();
			if (connect==null)
				throw new ConnectionException();
			
			Utility.executeOperation(connect, sql);
		}finally {
		//releases resources
		DBConnection.releaseConnection(connect);
		}
	}
	
	/**
	 * Delete a class from database
	 * @param Class to delete
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws MandatoryFieldException
	 * @throws InvalidValueException
	 */
	public synchronized void delete (Classroom pClassroom) throws ConnectionException, 
			SQLException, EntityNotFoundException, MandatoryFieldException, InvalidValueException {
		Connection connect = null;
		
		
		try {
			//ManagerUser.getInstance().userOnDeleteCascade(pUser);
			connect = DBConnection.getConnection();
				//Prepare SQL string
				String sql = "DELETE FROM " 
							+ ManagerClassroom.TABLE_CLASSROOM 
							+ " WHERE id_classroom = "
							+ Utility.isNull(pClassroom.getIdClassroom());
			
				Utility.executeOperation(connect, sql);
		}finally {
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}
	
	public synchronized Collection<Classroom> getClassroomsByStudent(User pUser) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException, MandatoryFieldException{
		Collection<Classroom> result=null;
		Connection connect = null;
		ManagerUser managerUser = ManagerUser.getInstance();
		try
		{
			// If no user exists
			if (!managerUser.exists(pUser))
					throw new EntityNotFoundException("The user does not exist!!!");
			if(!managerUser.isStudent(pUser))
					throw new InvalidValueException("The user is not a student!");
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			int iduser=managerUser.getUserId(pUser);
			String tSql = 
				
				"SELECT " 
				+ ManagerClassroom.TABLE_CLASSROOM 
				+".* FROM " 
				+ ManagerClassroom.TABLE_STUDENT_HAS_CLASSROOM 
				+ ", "
				+ ManagerClassroom.TABLE_CLASSROOM 
				+ " WHERE "
				+ ManagerClassroom.TABLE_STUDENT_HAS_CLASSROOM
				+ ".id_user = "  
				+ Utility.isNull(iduser)
				+" AND "
				+ ManagerClassroom.TABLE_CLASSROOM 
				+".id_classroom = "
				+ ManagerClassroom.TABLE_STUDENT_HAS_CLASSROOM 
				+".id_classroom";
				
				
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
				result = this.loadRecordsFromRs(tRs);
				
			if(result.isEmpty()) 
				throw new EntityNotFoundException("Unable to find Classes for the entered user");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	
	/**
	 * Returns the class that has the past ID
	 * @param The ID of the searched class
	 * @return a string representing the class with the supplied ID
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	public synchronized Classroom getClassroomByID(int pId) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException{
		Classroom result=null;
		Connection connect = null;
		try
		{
			// If the id has not been provided we return an error code
			if (pId <= 0)
				throw new EntityNotFoundException("Unable to find class!");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerClassroom.TABLE_CLASSROOM 
				+ " WHERE id_classroom = " 
				+ Utility.isNull(pId) ;
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			if (tRs.next()) 
				result = this.loadRecordFromRs(tRs);
			else 
				throw new EntityNotFoundException("Unable to find user!");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	
	
	/**
	 * Returns a collection of classes from the same academic year
	 */
	public synchronized Collection<Classroom> getClassroomsByAcademicYear(int pAcademicYear) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException{
		Collection<Classroom> result=null;
		Connection connect = null;
		try
		{
			// If the id has not been provided we return an error code
			if (pAcademicYear <= 1970)
				throw new EntityNotFoundException("Data troppo vecchia");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerClassroom.TABLE_CLASSROOM 
				+ " WHERE accademic_year = " 
				+ Utility.isNull(pAcademicYear) ;
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
				result = this.loadRecordsFromRs(tRs);
				
			if(result.isEmpty()) 
				throw new EntityNotFoundException("Unable to find Classes for date entered");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
		
	public synchronized Collection<Integer> getAcademicYearList() throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException{
		Collection<Integer> result=null;
		Connection connect = null;
		try
		{	
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT DISTINCT accademic_year FROM " 
				+ ManagerClassroom.TABLE_CLASSROOM
				+ " order by accademic_year ";
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
				result = this.loadIntegersFromRs(tRs);
				
			if(result.isEmpty()) 
				throw new EntityNotFoundException("Unable to find Classes for date entered");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	
	
	public synchronized Classroom getClassroomByUserAcademicYear(User pUser, int pAcademicYear) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException, MandatoryFieldException{
		Classroom result = null;
		Classroom temp = null;
		ManagerClassroom managerClassroom = ManagerClassroom.getInstance();
		Collection<Classroom> list = null;
		list = managerClassroom.getClassroomsByStudent(pUser);
		Iterator<Classroom> it = list.iterator();
		while(it.hasNext()){
			temp = it.next();
			if(temp.getAcademicYear()==pAcademicYear){
				result = temp;
				break;
			}
		}
		return result;
	}
	public synchronized Collection<Classroom> getClassroomsByTeacherAcademicYear(User pUser, int pAcademicYear) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException, MandatoryFieldException{
		Collection<Classroom> result = null;
		Connection connect = null;
		int idUser = pUser.getId();
		try
		{
			// If the id has not been provided we return an error code
			if (pAcademicYear <= 1970)
				throw new EntityNotFoundException("Data troppo vecchia");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * 
			 */
			String tSql = 
				"SELECT DISTINCT " 
				+ ManagerClassroom.TABLE_CLASSROOM +".* FROM "  
				+ ManagerClassroom.TABLE_CLASSROOM + ", "
				+ ManagerClassroom.TABLE_TEACHER_HAS_CLASSROOM 
				+ " WHERE  "
				+ ManagerClassroom.TABLE_CLASSROOM + ".id_classroom = "
				+ ManagerClassroom.TABLE_TEACHER_HAS_CLASSROOM 
				+ ".id_classroom  AND "
				+ ManagerClassroom.TABLE_CLASSROOM + ".accademic_year = "
				+ Utility.isNull(pAcademicYear)
				+ " AND "
				+ ManagerClassroom.TABLE_TEACHER_HAS_CLASSROOM + ".id_user = "
				+ Utility.isNull(idUser)
				;
			
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
				result = this.loadRecordsFromRs(tRs);
				
			if(result.isEmpty()) 
				throw new EntityNotFoundException("Unable to find Classes for user and year entered");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	
	}
	public synchronized Collection<Classroom> getClassroomsByTeacher(User pUser) 
	throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException, MandatoryFieldException{
		Collection<Classroom> result=null;
		Connection connect = null;
		ManagerUser managerUser = ManagerUser.getInstance();
		try
		{
			// If no user exists
			if (!managerUser.exists(pUser))
					throw new EntityNotFoundException("The user does not exist!!!");
			if(!managerUser.isTeacher(pUser))
					throw new InvalidValueException("The user is not a student!");
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			int iduser=managerUser.getUserId(pUser);
			String tSql = 
				
				"SELECT DISTINCT " 
				+ ManagerClassroom.TABLE_CLASSROOM 
				+".* FROM " 
				+ ManagerClassroom.TABLE_TEACHER_HAS_CLASSROOM 
				+ ", "
				+ ManagerClassroom.TABLE_CLASSROOM 
				+ " WHERE "
				+ ManagerClassroom.TABLE_TEACHER_HAS_CLASSROOM
				+ ".id_user = "  
				+ Utility.isNull(iduser)
				+" AND "
				+ ManagerClassroom.TABLE_CLASSROOM 
				+".id_classroom = "
				+ ManagerClassroom.TABLE_TEACHER_HAS_CLASSROOM 
				+".id_classroom";
				
				
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
				result = this.loadRecordsFromRs(tRs);
				
			if(result.isEmpty()) {
				
				throw new EntityNotFoundException("Unable to find Classes for the entered user");
			}
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	/** Consistent reading an integer from recod resultSet
	 * 
	 * @param pRs
	 * 		resultSet
	 * @return
	 * 	collection<Integer>
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	private Collection<Integer> loadIntegersFromRs(ResultSet pRs) throws SQLException, InvalidValueException{
		Collection<Integer> result = new Vector<Integer>();
		while(pRs.next())  {
			result.add(pRs.getInt("accademic_year"));
		} 
		return result;
	}



	/**
	 * Allows you to read only one record from the Result Set
	 * @param The result set from which to extract the Classroom object
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	private Classroom loadRecordFromRs(ResultSet pRs) throws SQLException, InvalidValueException{
		Classroom classroom = new Classroom();
		classroom.setName(pRs.getString("name"));
		classroom.setAcademicYear(pRs.getInt("accademic_year"));
		classroom.setIdClassroom(pRs.getInt("id_classroom"));
		classroom.setIdAddress(pRs.getInt("id_address"));
		return classroom;
	}

	/**
	 * Lets you read multiple records from the Result Set
	 * @param The result set from which to extract the Classroom object
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	
	private Collection<Classroom> loadRecordsFromRs(ResultSet pRs) throws SQLException, InvalidValueException{
		Collection<Classroom> result = new Vector<Classroom>();
		while(pRs.next())  {
			result.add(loadRecordFromRs(pRs));
		} 
		return result;
	}

}
