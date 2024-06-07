package smos.storage;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Date;
import java.util.Vector;

import smos.bean.Absence;
import smos.bean.Delay;
import smos.bean.Justify;
import smos.bean.Note;
import smos.bean.RegisterLine;
import smos.bean.UserListItem;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.exception.MandatoryFieldException;
import smos.storage.connectionManagement.DBConnection;
import smos.storage.connectionManagement.exception.ConnectionException;
import smos.utility.Utility;

public class ManagerRegister {

	
	/**
	 * Class that manages the Digital Registry
	 * @author Nicola Pisanti
	 * @version 1.0
	 */
	
	private static ManagerRegister instance;
	
	public final static String TABLE_ABSENCE="absence";
	public final static String TABLE_DELAY="delay";
	public final static String TABLE_JUSTIFY="justify";
	public final static String TABLE_NOTE="note";
	
	
	private ManagerRegister(){
		super();
	}
	
	
	/**
	 * Returns the only instance of the existing class.
	 *
	 * @return Returns the instance of the class.
	 */
	public static synchronized ManagerRegister getInstance(){
		if(instance==null){
			instance = new ManagerRegister();
		}
		return instance;
	}
	
	/**
	 * Check if the input given class is in the database
	 * @param pAbsence
	 * 		The class whose existence must be verified
	 * @return true if the class is in the database, otherwise false
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	public synchronized boolean exists(Absence pAbsence) throws ConnectionException, SQLException {
		
		boolean result = false;
		Connection connect = null;

		try {
			//We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			String sql = "SELECT * FROM " 
				+ ManagerRegister.TABLE_ABSENCE
				+ " WHERE id_absence = "
				+ Utility.isNull(pAbsence.getIdAbsence());

			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, sql);

			if (tRs.next()){
				result = true;
			}
			
			return result;
			
		} finally {
			DBConnection.releaseConnection(connect);
		}
	}
	
	/**
	 * Check if the input given class is in the database
	 * @param pDelay
	 * 		The class whose existence must be verified
	 * @return true if the class is in the database, otherwise false
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	public synchronized boolean exists(Delay pDelay) throws ConnectionException, SQLException {
		
		boolean result = false;
		Connection connect = null;

		try {
			//We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			String sql = "SELECT * FROM " 
				+ ManagerRegister.TABLE_ABSENCE
				+ " WHERE id_delay = "
				+ Utility.isNull(pDelay.getIdDelay());
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
	 * Check if the input given class is in the database
	 * @param pDelay
	 * 		The class whose existence must be verified
	 * @return true if the class is in the database, otherwise false
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	public synchronized boolean exists(Justify pJustify) throws ConnectionException, SQLException {
		
		boolean result = false;
		Connection connect = null;

		try {
			//We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			String sql = "SELECT * FROM " 
				+ ManagerRegister.TABLE_JUSTIFY
				+ " WHERE  id_justify = "
				+ Utility.isNull(pJustify.getIdJustify());

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
	 * Inserts an absence in the database
	 * @param pAbsence
	 * 		an object of type Absence to enter in the database
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void insertAbsence(Absence pAbsence) throws  
		ConnectionException, SQLException, EntityNotFoundException, InvalidValueException{
		
		Connection connect= null;
		try{
			
			connect = DBConnection.getConnection();
			if (connect==null)
				throw new ConnectionException();
			//Prepare Sql string
			String sql =
				"INSERT INTO " 
				+ ManagerRegister.TABLE_ABSENCE 
				+ " (id_user, date_absence, id_justify, accademic_year) " 
				+ "VALUES (" 
				+ Utility.isNull(pAbsence.getIdUser()) 
				+ "," 
				+ Utility.isNull(pAbsence.getDateAbsence()) 
				+ "," 
				+ Utility.isNull(pAbsence.getIdJustify()) 
				+ "," 
				+ Utility.isNull(pAbsence.getAcademicYear())
				+ ")";
		
			Utility.executeOperation(connect,sql);
		
			pAbsence.setIdAbsence((Utility.getMaxValue("id_absence",ManagerRegister.TABLE_ABSENCE)));
		
		}finally {
		//releases resources
		
		DBConnection.releaseConnection(connect);
		}
	}
	

	/**
	 * Inserts a delay in the database
	 * @param pDelay
	 * 		a Delay object to enter in the database
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void insertDelay(Delay pDelay) throws  
		ConnectionException, SQLException, EntityNotFoundException, InvalidValueException{
		
		Connection connect= null;
		try{
			
			connect = DBConnection.getConnection();
			if (connect==null)
				throw new ConnectionException();
			//Prepare Sql string
			String sql =
				"INSERT INTO " 
				+ ManagerRegister.TABLE_DELAY 
				+ " (id_user, date_delay, time_delay, accademic_year) " 
				+ "VALUES (" 
				+ Utility.isNull(pDelay.getIdUser()) 
				+ "," 
				+ Utility.isNull(pDelay.getDateDelay()) 
				+ "," 
				+ Utility.isNull(pDelay.getTimeDelay()) 
				+ "," 
				+ Utility.isNull(pDelay.getAcademicYear())
				+ ")";
		
			Utility.executeOperation(connect,sql);
		
			pDelay.setIdDelay((Utility.getMaxValue("id_delay",ManagerRegister.TABLE_DELAY)));
		
		}finally {
		//Prepare Sql string
		
		DBConnection.releaseConnection(connect);
		}
	}
	


	/**
	 * Prepare Sql string
	 * @param pNote
	 * 		an object of type Notes to insert in the database
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void insertNote(Note pNote) throws MandatoryFieldException,  
		ConnectionException, SQLException, EntityNotFoundException, InvalidValueException{
		
		Connection connect= null;
		try{
			if (pNote.getDescription() == null || pNote.getDescription().equals(""))
				throw new MandatoryFieldException("an object of type Notes to insert in the database");
			
			if (pNote.getTeacher() == null || pNote.getTeacher().equals("") )
				throw new MandatoryFieldException("Insert the teacher");
			
			connect = DBConnection.getConnection();
			if (connect==null)
				throw new ConnectionException();
			//Prepare Sql string
			String sql =
				"INSERT INTO " 
				+ ManagerRegister.TABLE_NOTE 
				+ " (id_user, date_note, description, teacher, accademic_year) " 
				+ "VALUES (" 
				+ Utility.isNull(pNote.getIdUser()) 
				+ "," 
				+ Utility.isNull(pNote.getDateNote()) 
				+ "," 
				+ Utility.isNull(pNote.getDescription()) 
				+ "," 
				+ Utility.isNull(pNote.getTeacher()) 
				+ "," 
				+ Utility.isNull(pNote.getAcademicYear())
				+ ")";
		
			Utility.executeOperation(connect,sql);
		
			pNote.setIdNote((Utility.getMaxValue("id_note",ManagerRegister.TABLE_NOTE)));
		
		}finally {
		//releases resources
		
		DBConnection.releaseConnection(connect);
		}
	}
	

	/**
	 * Inserts a justification in the database
	 * @param pJustify 
	 * 		an object of type Justify to enter in the database
	 * @param pAbsence
	 * 		an object of type Absence representing the justified absence
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void insertJustify(Justify pJustify, Absence pAbsence) throws   
		ConnectionException, SQLException, EntityNotFoundException, InvalidValueException{
		
		Connection connect= null;
		try{
			
			
			connect = DBConnection.getConnection();
			if (connect==null)
				throw new ConnectionException();
			//Prepare Sql string
			String sql =
				"INSERT INTO " 
				+ ManagerRegister.TABLE_JUSTIFY 
				+ " (id_user, date_justify, accademic_year) " 
				+ "VALUES (" 
				+ Utility.isNull(pJustify.getIdUser()) 
				+ "," 
				+ Utility.isNull(pJustify.getDateJustify()) 
				+ "," 
				+ Utility.isNull(pJustify.getAcademicYear())
				+ ")";
		
			Utility.executeOperation(connect,sql);
		
			pJustify.setIdJustify((Utility.getMaxValue("id_justify",ManagerRegister.TABLE_JUSTIFY)));
			
			pAbsence.setIdJustify(pJustify.getIdJustify());
			this.updateAbsence(pAbsence);
		
		}finally {
		//Prepare Sql string
		
		DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Delete an absence from database
	 * @param pAbsence
	 * 		the absence to be deleted
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void deleteAbsence (Absence pAbsence) throws ConnectionException, 
			SQLException, EntityNotFoundException, MandatoryFieldException, InvalidValueException {
		Connection connect = null;
		
		
		try {
			connect = DBConnection.getConnection();
				//Prepare SQL string
				String sql = "DELETE FROM " 
							+ ManagerRegister.TABLE_ABSENCE 
							+ " WHERE id_absence = "
							+ Utility.isNull(pAbsence.getIdAbsence());
			
				Utility.executeOperation(connect, sql);
				
				if (!(pAbsence.getIdJustify()==null)){
					deleteJustify(pAbsence.getIdJustify());
				}
		}finally {
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}


	/**
	 * Delete a delay from database
	 * @param pDelay
	 * 		the delay to be cancelled
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void deleteDelay (Delay pDelay) throws ConnectionException, 
			SQLException, EntityNotFoundException, MandatoryFieldException, InvalidValueException {
		Connection connect = null;
		
		
		try {
			connect = DBConnection.getConnection();
				//Prepare SQL string
				String sql = "DELETE FROM " 
							+ ManagerRegister.TABLE_DELAY 
							+ " WHERE id_delay = "
							+ Utility.isNull(pDelay.getIdDelay());
			
				Utility.executeOperation(connect, sql);
		}finally {
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Delete a note from database
	 * @param pNote
	 * 		the note to be deleted
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void deleteNote (Note pNote) throws ConnectionException, 
			SQLException, EntityNotFoundException, InvalidValueException {
		Connection connect = null;
		
		
		try {
			connect = DBConnection.getConnection();
				//Prepare SQL string
				String sql = "DELETE FROM " 
							+ ManagerRegister.TABLE_NOTE 
							+ " WHERE id_note = "
							+ Utility.isNull(pNote.getIdNote());
			
				Utility.executeOperation(connect, sql);
		}finally {
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Delete a note from database
	 * @param pJIDustify
	 * 		the ID of the note to be deleted
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void deleteJustify (int pIDJustify) throws ConnectionException, 
			SQLException, EntityNotFoundException, MandatoryFieldException, InvalidValueException {
		Connection connect = null;
		
		
		try {
			connect = DBConnection.getConnection();
				//Prepare SQL string
				String sql = "DELETE FROM " 
							+ ManagerRegister.TABLE_JUSTIFY 
							+ " WHERE id_justify = "
							+ Utility.isNull(pIDJustify);
			
				Utility.executeOperation(connect, sql);
				
				try{
					Absence temp= getAbsenceByIdJustify(pIDJustify);
					temp.setIdJustify(0);
					updateAbsence(temp);
				}catch(Exception e){
					// Normal if an exception is generated
					   Since we may be deleting a justification
					// whose absence we just cancelled
				}
				
				
		}finally {
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Update the statistics of an absence
	 * @param pAbsence
	 * 		Absence with updated statistics (but identical ID)
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws MandatoryFieldException
	 */
	
	
	public synchronized void updateAbsence (Absence pAbsence) throws ConnectionException,
	SQLException, EntityNotFoundException{
		Connection connect= null;
		
		try{

			
			//Prepare SQL string
			String sql=
				"UPDATE " 
				+	ManagerRegister.TABLE_ABSENCE 
				+ " SET" 
				+ " id_user = " 
				+ Utility.isNull(pAbsence.getIdUser()) 
				+ ", date_absence = " 
				+ Utility.isNull(pAbsence.getDateAbsence()) 
				+ ", id_justify = " 
				+ Utility.isNull(pAbsence.getIdJustify())  
				+ ", accademic_year = " 
				+ Utility.isNull(pAbsence.getAcademicYear())  
				+ " WHERE id_absence = " 
				+ Utility.isNull(pAbsence.getIdAbsence());
			
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
	 * Update the statistics of a delay
	 * @param pDelay
	 * 		The delay with updated statistics (but identical ID)
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws MandatoryFieldException
	 */
	
	
	public synchronized void updateDelay (Delay pDelay) throws ConnectionException,
	SQLException, EntityNotFoundException, MandatoryFieldException{
		Connection connect= null;
		
		try{

			
			//Prepare SQL string
			String sql=
				"UPDATE " 
				+	ManagerRegister.TABLE_DELAY 
				+ " SET" 
				+ " id_user = " 
				+ Utility.isNull(pDelay.getIdUser()) 
				+ ", date_delay = " 
				+ Utility.isNull(pDelay.getDateDelay()) 
				+ ", time_delay = " 
				+ Utility.isNull(pDelay.getTimeDelay())  
				+ ", accademic_year = " 
				+ Utility.isNull(pDelay.getAcademicYear())  
				+ " WHERE id_delay = " 
				+ Utility.isNull(pDelay.getIdDelay());
			
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
	 * Method that updates the statistics of a Note
	 * @param pNote
	 * 		an object of type Notes with updated statistics but identical id
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws MandatoryFieldException
	 */
	
	public synchronized void updateNote (Note pNote) throws ConnectionException,
	SQLException, EntityNotFoundException, MandatoryFieldException{
		Connection connect= null;
		
		try{
			if (pNote.getDescription() == null || pNote.getDescription().equals(""))
				throw new MandatoryFieldException("Insert note text");
			
			if (pNote.getTeacher() == null || pNote.getTeacher().equals("") )
				throw new MandatoryFieldException("Insert the teacher");
			//Prepare SQL string
			String sql=
				"UPDATE " 
				+	ManagerRegister.TABLE_NOTE
				+ " SET" 
				+ " id_user = " 
				+ Utility.isNull(pNote.getIdUser()) 
				+ ", date_note = " 
				+ Utility.isNull(pNote.getDateNote())   
				+ ", description = " 
				+ Utility.isNull(pNote.getDescription())   
				+ ", teacher = " 
				+ Utility.isNull(pNote.getTeacher())   
				+ ", accademic_year = " 
				+ Utility.isNull(pNote.getAcademicYear())  
				+ " WHERE id_note = " 
				+ Utility.isNull(pNote.getIdNote());
			
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
	 * Update statistics of a justifier
	 * @param pJustify
	 * 		justify it with updated statistics (but identical ID)
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws MandatoryFieldException
	 */
	

	public synchronized void updateJustify (Justify pJustify) throws ConnectionException,
	SQLException, EntityNotFoundException, MandatoryFieldException{
		Connection connect= null;
		
		try{

			
			//Prepare SQL string
			String sql=
				"UPDATE " 
				+	ManagerRegister.TABLE_JUSTIFY
				+ " SET" 
				+ " id_user = " 
				+ Utility.isNull(pJustify.getIdUser()) 
				+ ", date_justify = " 
				+ Utility.isNull(pJustify.getDateJustify())   
				+ ", accademic_year = " 
				+ Utility.isNull(pJustify.getAcademicYear())  
				+ " WHERE id_justify = " 
				+ Utility.isNull(pJustify.getIdJustify());
			
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
	 * Method that returns a note given the id of the note itself
	 * @param pIDJustify
	 * 		an integer representing the id of the note 
	 * @return a Notes object representing the note
	 * @throws InvalidValueException
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	
	
	
	public synchronized Note getNoteById( int pIDNote)throws InvalidValueException,
			EntityNotFoundException, ConnectionException, SQLException{
		Note result=null;
		Connection connect = null;
		try
		{
			// If the id has not been provided we return an error code
			if (pIDNote<=0)
				throw new EntityNotFoundException("Unable to find note");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerRegister.TABLE_NOTE 
				+ " WHERE id_note = " 
				+ Utility.isNull(pIDNote) ;
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			if (tRs.next()) 
				result = this.loadNoteFromRs(tRs);
			else 
				throw new EntityNotFoundException("Unable to find the note!");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}	
	
	
	
	
	
	
	
	/**
	 * Method that returns an absence given the id of justifca associated with that absence
	 * @param pIDJustify
	 * 		an integer representing the id of the justify
	 * @return an object of type Absence representing the justified absence
	 * @throws InvalidValueException
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	
	
	
	public synchronized Absence getAbsenceByIdJustify( int pIDJustify)throws InvalidValueException,
			EntityNotFoundException, ConnectionException, SQLException{
		Absence result=null;
		Connection connect = null;
		try
		{
			// If the id has not been provided we return an error code
			if (pIDJustify<=0)
				throw new EntityNotFoundException("Unable to find absence");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerRegister.TABLE_ABSENCE 
				+ " WHERE id_justify = " 
				+ Utility.isNull(pIDJustify) ;
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			if (tRs.next()) 
				result = this.loadAbsenceFromRs(tRs);
			else 
				throw new EntityNotFoundException("Unable to find absence!");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	/**
	 * Method that returns an absence given the id of this
	 * @param pIDAbsence
	 * 		an integer representing the id of the absence
	 * @return an object of type Absence representing the absence
	 * @throws InvalidValueException
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	
	
	
	public synchronized Absence getAbsenceByIdAbsence( int pIDAbsence)throws InvalidValueException,
			EntityNotFoundException, ConnectionException, SQLException{
		Absence result=null;
		Connection connect = null;
		try
		{
			// If the id has not been provided we return an error code
			if (pIDAbsence<=0)
				throw new EntityNotFoundException("Unable to find absence");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerRegister.TABLE_ABSENCE 
				+ " WHERE id_absence = " 
				+ Utility.isNull(pIDAbsence) ;
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			if (tRs.next()) 
				result = this.loadAbsenceFromRs(tRs);
			else 
				throw new EntityNotFoundException("Unable to find absence!");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	
	
	/**
	 * Method that returns a delay given the id of this
	 * @param pIDDelay
	 * 		an integer representing the delay id
	 * @return a Delay object representing the delay
	 * @throws InvalidValueException
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	
	
	
	public synchronized Delay getDelayById( int pIDDelay)throws InvalidValueException,
			EntityNotFoundException, ConnectionException, SQLException{
		Delay result=null;
		Connection connect = null;
		try
		{
			// If the id has not been provided we return an error code
			if (pIDDelay<=0)
				throw new EntityNotFoundException("Unable to find ritardo");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerRegister.TABLE_DELAY
				+ " WHERE id_delay = " 
				+ Utility.isNull(pIDDelay) ;
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			if (tRs.next()) 
				result = this.loadDelayFromRs(tRs);
			else 
				throw new EntityNotFoundException("Unable to find absence!");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	/**
	 * Method that returns an absence given the id of this
	 * @param pIDAbsence
	 * 		an integer representing the id of the absence
	 * @return an object of type Absence representing the absence
	 * @throws InvalidValueException
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	
	
	
	public synchronized Justify getJustifyByIdJustify( int pIDJustify)throws InvalidValueException,
			EntityNotFoundException, ConnectionException, SQLException{
		Justify result=null;
		Connection connect = null;
		try
		{
			// If the id has not been provided we return an error code
			if (pIDJustify<=0)
				throw new EntityNotFoundException("Unable to find justification");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerRegister.TABLE_JUSTIFY 
				+ " WHERE id_justify = " 
				+ Utility.isNull(pIDJustify) ;
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			if (tRs.next()) 
				
				result= this.loadJustifyFromRs(tRs);
			else 
				throw new EntityNotFoundException("Impossible to find justification!");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	/**
	 * Method that returns true if the absence given in input has an assigned justification
	 * @param pAbsence
	 * 		an Absence value object of which you have to check if it justifies
	 * @return true if absence is justified, false otherwise
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	
	
	public synchronized boolean hasJustify(Absence pAbsence)throws EntityNotFoundException, ConnectionException, SQLException{
		if(!exists(pAbsence)) throw new EntityNotFoundException("Absence not present in database");
		if(pAbsence.getIdJustify()==null) return false;
		return true;
	}
	
	/**
	 * Method that returns the justification linked to a given absence
	 * @param pAbsence
	 * 		an object of type Absence representing the absence
	 * @return	an object of type Justify, or null if the absence does not justify
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */

	public synchronized Justify getJustifyByAbsence(Absence pAbsence)throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException{
		if(!exists(pAbsence)) throw new EntityNotFoundException("Absence not present in database");
		if(pAbsence.getIdJustify()==null) return null;
		
		Justify result=null;
		Connection connect = null;
		try{
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerRegister.TABLE_JUSTIFY 
				+ " WHERE id_justify = " 
				+ Utility.isNull(pAbsence.getIdJustify()) ;
			
			//We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			if (tRs.next()) 
				result = this.loadJustifyFromRs(tRs);
			else 
				throw new EntityNotFoundException("Impossible to find justification!");
			
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
		
		
	}

	/**
	 * Method that returns absences taken a given school year and user input
	 * @param pIdUser
	 * 		an integer representing the user id
	 * @param pAcademicYear
	 * 		a whole representing the academic year
	 * @return a collection of absences (empty if the user did not have absences )
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	
	
	public synchronized Collection<Absence> getAbsenceByIDUserAndAcademicYear(int pIdUser, int pAcademicYear) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException{
		Collection<Absence> result=new Vector<Absence>();
		Connection connect = null;
		try
		{
			// If the id has not been provided we return an error code
			if (pAcademicYear <= 1970)
				throw new EntityNotFoundException("Date too old");

			// idem per l'id user
			if (pIdUser<=0)
				throw new EntityNotFoundException("User not found");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerRegister.TABLE_ABSENCE 
				+ " WHERE accademic_year = " 
				+ Utility.isNull(pAcademicYear) 
				+ " AND id_user = "
				+ Utility.isNull(pIdUser);
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			while(tRs.next())  {
				result.add(loadAbsenceFromRs(tRs));
			} 
				
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Method that returns a collection of notes for a given user and a given school year
	 * @param pIdUser
	 * 		an integer representing the user id
	 * @param pAcademicYear
	 * 		a whole representing the academic year
	 * @return a collection of notes, empty if the user has not received any
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	public synchronized Collection<Note> getNoteByIDUserAndAcademicYear(int pIdUser, int pAcademicYear) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException{
		Collection<Note> result=new Vector<Note>();
		Connection connect = null;
		try
		{
			//If the id has not been provided we return an error code
			if (pAcademicYear <= 1970)
				throw new EntityNotFoundException("Too old date");

			// idem for user id
			if (pIdUser<=0)
				throw new EntityNotFoundException("User not found");
			
			
			/*
			 *Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerRegister.TABLE_NOTE 
				+ " WHERE accademic_year = " 
				+ Utility.isNull(pAcademicYear) 
				+ " AND id_user = "
				+ Utility.isNull(pIdUser);
			
			//We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			while(tRs.next())  {
				result.add(loadNoteFromRs(tRs));
			} 
				
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	
	/**
	 * Method that returns the absence of a given student on a given day
	 * @param pIdUser
	 * 		an integer representing the student id
	 * @param pDate
	 * 		a string representing the formatted date for the database
	 * @return un oggetto di tipo Absence, oppure null se lo studente era presente
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */

	
	public synchronized Absence getAbsenceByIDUserAndDate(int pIdUser, Date pDate) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException{
		Absence result=new Absence();
		Connection connect = null;
		try
		{
			//TODO String Formatting Controls
			
			
			// idem for user id
			if (pIdUser<=0)
				throw new EntityNotFoundException("User not found");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerRegister.TABLE_ABSENCE 
				+ " WHERE date_absence = " 
				+ Utility.isNull(pDate) 
				+ " AND id_user = "
				+ Utility.isNull(pIdUser);
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			if(tRs.next())  {
				result=loadAbsenceFromRs(tRs);
			}else {
				result=null;
			}
				
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	
	
	/**
	 * Method that returns the delay of a given student on a given day
	 * @param pIdUser
	 * 		an integer representing the student id
	 * @param pDate
	 * 		a string representing the formatted date for the database
	 * @return a Delay object, or null if the student was on time or absent
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */

	
	public synchronized Delay getDelayByIDUserAndDate(int pIdUser, Date pDate) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException{
		Delay result=new Delay();
		Connection connect = null;
		try
		{
			//TODO String Formatting Controls
			
			
			// idem for user id
			if (pIdUser<=0)
				throw new EntityNotFoundException("User not found");
			
			
			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the class of the past id
			 */
			String tSql = 
				"SELECT * FROM " 
				+ ManagerRegister.TABLE_DELAY 
				+ " WHERE date_delay = " 
				+ Utility.isNull(pDate) 
				+ " AND id_user = "
				+ Utility.isNull(pIdUser);
			
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null) 
				throw new ConnectionException();
			
			
			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);
			
			if(tRs.next())  {
				result=loadDelayFromRs(tRs);
			}else {
				result=null;
			}
				
			return result;
		}finally{
			DBConnection.releaseConnection(connect);
		}
	}
	
	
	public synchronized Collection<RegisterLine> getRegisterByClassIDAndDate(int pClassID, Date pDate) throws EntityNotFoundException, ConnectionException, SQLException, InvalidValueException{
		
		Collection<RegisterLine> result = new Vector<RegisterLine>();
		ManagerUser mg = ManagerUser.getInstance();
		
		Collection<UserListItem> students = mg.getStudentsByClassroomId(pClassID);
		
		
		for (UserListItem x : students){
			RegisterLine temp = new RegisterLine();
			temp.setStudent(x);
			temp.setAbsence(this.getAbsenceByIDUserAndDate(x.getId(), pDate));
			temp.setDelay(this.getDelayByIDUserAndDate(x.getId(), pDate));
			result.add(temp);
		}
		
		return result;
	}

	/**
	 * Method to check if there is an absence in a registry line
	 * @param pRegisterLine
	 * 		an object of type RegisterLine
	 * @return	true if there is an absence in the past log line, otherwise false
	*/
	
	
	public boolean hasAbsence(RegisterLine pRegisterLine){
		if(pRegisterLine.getAbsence()==null)return false;
		return true;
	}

	/**
	 * Method to check if there is a delay in a log line
	 * @param pRegisterLine
	 * 		an object of type RegisterLine
	 * @return	true if there is a delay in the past log line, otherwise false
	 */
	
	
	public boolean hasDelay(RegisterLine pRegisterLine){
		if(pRegisterLine.getDelay()==null)return false;
		return true;
	}

	/**
	 * Allows you to read only one record from the Result Set
	 * @param pRs
	 * 		The result set from which to extract the Absence object
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	private Absence loadAbsenceFromRs(ResultSet pRs) throws SQLException, InvalidValueException{
		Absence absence = new Absence();
		
		absence.setIdAbsence(pRs.getInt("id_absence"));
		absence.setIdUser(pRs.getInt("id_user"));
		absence.setDateAbsence((Date)pRs.getDate("date_absence"));
		absence.setIdJustify(pRs.getInt("id_justify"));
		absence.setAcademicYear(pRs.getInt("accademic_year"));
		
		return absence;
	}
	
	/**
	 * Allows you to read only one record from the Result Set
	 * @param pRs
	 * 		The result set from which to extract the Justify object
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	private Justify loadJustifyFromRs(ResultSet pRs) throws SQLException, InvalidValueException{
		Justify justify = new Justify();
		
		justify.setIdJustify(pRs.getInt("id_justify"));
		justify.setIdUser(pRs.getInt("id_user"));
		justify.setDateJustify((Date)pRs.getDate("date_justify"));
		justify.setAcademicYear(pRs.getInt("accademic_year"));
		
		return justify;
	}
	/**
	 * Allows you to read only one record from the Result Set
	 * @param pRs
	 * 		The result set from which to extract the Notes object
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	
	private Note loadNoteFromRs(ResultSet pRs) throws SQLException, InvalidValueException{
		Note note= new Note();
		
		note.setIdNote(pRs.getInt("id_note"));
		note.setIdUser(pRs.getInt("id_user"));
		note.setDateNote((Date)pRs.getDate("date_note"));
		note.setDescription(pRs.getString("description"));
		note.setTeacher(pRs.getString("teacher"));	
		note.setAcademicYear(pRs.getInt("accademic_year"));
	
		return note;
	}

	/**
	 * Allows you to read only one record from the Result Set
	 * @param pRs
	 * 		The result set from which to extract the Delay object
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	
	private Delay loadDelayFromRs(ResultSet pRs) throws SQLException, InvalidValueException{
		Delay delay = new Delay();
				
		delay.setIdDelay(pRs.getInt("id_delay"));
		delay.setIdUser(pRs.getInt("id_user"));
		delay.setDateDelay((Date)pRs.getDate("date_delay"));
		delay.setTimeDelay(pRs.getString("time_delay"));
		delay.setAcademicYear(pRs.getInt("accademic_year"));
	
		return delay;
	}
}
