package smos.storage;

import smos.bean.Teaching;
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
 * Teaching manager class.
 * 
 * @author Giulio D'Amora
 * @version 1.0
 * 
 *          2009 � Copyright by SMOS
 */

public class ManagerTeaching {
	private static ManagerTeaching instance;

	/**
	 * The name of the teaching table.
	 */
	public static final String TABLE_TEACHING = "teaching";

	/**
	 * The name of the table that shapes the association many to many between
	 * addresses and teachings.
	 */
	public static final String TABLE_ADDRESS_TEACHINGS = "address_has_teaching";

	/**
	 * The name of the table that models the association many to many among users
	 * and teachings.
	 */
	public static final String TABLE_TEACHER_CLASSROOM = "teacher_has_classroom";

	/**
	 * The class builder.
	 */
	private ManagerTeaching() {
		super();
	}

	/**
	 * The only instance of existing teaching returns.
	 * 
	 * @return Returns the instance of the class.
	 */
	public static synchronized ManagerTeaching getInstance() {
		if (instance == null) {
			instance = new ManagerTeaching();
		}
		return instance;
	}

	/**
	 * Verify the existence of a teaching in the database.
	 * 
	 * @param pTeaching
	 *            The teaching to control.
	 * @return Returns true if there is past teaching as a parameter,
	 *         otherwise false.
	 * 
	 * @throws MandatoryFieldException
	 * @throws SQLException
	 * @throws ConnectionException
	 */
	public synchronized boolean exists(Teaching pTeaching)
			throws MandatoryFieldException, ConnectionException, SQLException {

		boolean result = false;
		Connection connect = null;

		if (pTeaching.getName() == null)
			throw new MandatoryFieldException("Specify the name.");
		try {
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			String sql = "SELECT * FROM " + ManagerTeaching.TABLE_TEACHING
					+ " WHERE name = " + Utility.isNull(pTeaching.getName());

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
	 * Inserts a new teaching in the teaching table.
	 * 
	 * @param pTeaching
	 *            The teaching to be included.
	 * 
	 * @throws SQLException
	 * @throws ConnectionException
	 * @throws MandatoryFieldException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void insert(Teaching pTeaching)
			throws MandatoryFieldException, ConnectionException, SQLException,
			EntityNotFoundException, InvalidValueException {
		Connection connect = null;
		try {
			// control of mandatory fields
			if (pTeaching.getName() == null)
				throw new MandatoryFieldException("Specify the name field");

			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();
			// Prepare Sql string
			String sql = "INSERT INTO " + ManagerTeaching.TABLE_TEACHING
					+ " (name) " + "VALUES ("
					+ Utility.isNull(pTeaching.getName()) + ")";

			Utility.executeOperation(connect, sql);

			pTeaching.setId(Utility.getMaxValue("id_teaching",
					ManagerTeaching.TABLE_TEACHING));

		} finally {
			// releases resources

			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Update a teaching in the teaching table.
	 *
	 * @param pTeaching
	 *            Teaching to be modified
	 * 
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws MandatoryFieldException
	 */
	public synchronized void update(Teaching pTeaching)
			throws ConnectionException, SQLException, EntityNotFoundException,
			MandatoryFieldException {
		Connection connect = null;

		try {
			if (pTeaching.getId() <= 0)
				throw new EntityNotFoundException(
						"Can't find the teaching!");

			if (pTeaching.getName() == null)
				throw new MandatoryFieldException("Specify the name field");

			// Prepare SQL string
			String sql = "UPDATE " + ManagerTeaching.TABLE_TEACHING + " SET"
					+ " name = " + Utility.isNull(pTeaching.getName())
					+ " WHERE id_teaching = "
					+ Utility.isNull(pTeaching.getId());

			// make a new connection and send the query
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			Utility.executeOperation(connect, sql);
		} finally {
			// releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Delete a teaching from the teaching table.
	 * 
	 * @param pTeaching
	 *            The teaching to be eliminated.
	 * 
	 * @throws MandatoryFieldException
	 * @throws EntityNotFoundException
	 * @throws SQLException
	 * @throws ConnectionException
	 * @throws InvalidValueException
	 * 
	 */
	public synchronized void delete(Teaching pTeaching)
			throws ConnectionException, SQLException, EntityNotFoundException,
			MandatoryFieldException, InvalidValueException {
		Connection connect = null;

		try {
			// ManagerTeaching.getInstance().teachingOnDeleteCascade(pTeaching);
			connect = DBConnection.getConnection();
			// Prepare SQL string
			String sql = "DELETE FROM " + ManagerTeaching.TABLE_TEACHING
					+ " WHERE id_teaching = "
					+ Utility.isNull(pTeaching.getId());

			Utility.executeOperation(connect, sql);
		} finally {
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Returns the id of the past teaching as a parameter.
	 * 
	 * @param pTeaching
	 *            The teaching of which the id is required.
	 * @return Returns the id of the past teaching as a parameter.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	public synchronized int getTeachingId(Teaching pTeaching)
			throws EntityNotFoundException, ConnectionException, SQLException {
		int result = 0;
		Connection connect = null;
		try {
			if (pTeaching == null)
				throw new EntityNotFoundException(
						"Unable to find teaching!");

			/*
			 *Preparing SQL string to retrieve information
			 * corresponding to the id of the teaching passed as a parameter.
			 */
			String tSql = "SELECT id_teaching FROM "
					+ ManagerTeaching.TABLE_TEACHING + " WHERE name = "
					+ Utility.isNull(pTeaching.getName());

			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			//Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);

			if (tRs.next())
				result = tRs.getInt("id_teaching");

			return result;
		} finally {
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Returns the name of the teaching corresponding to the past id as
	 * parameter.
	 * 
	 * @param pId
	 *            The teaching id.
	 * @return Returns a string containing the name of the teaching.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	public synchronized String getTeachingNameById(int pId)
			throws EntityNotFoundException, ConnectionException, SQLException {
		String result;
		Connection connect = null;
		try {
			// If the id has not been provided we return an error code
			if (pId <= 0)
				throw new EntityNotFoundException(
						"Can't find the teaching!");

			/*
			 * Preparing SQL string to retrieve information
			 * corresponding to the id of the past teaching as a parameter
			 */
			String tSql = "SELECT name FROM " + ManagerTeaching.TABLE_TEACHING
					+ " WHERE id_teaching = " + Utility.isNull(pId);

			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			//Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);

			if (tRs.next())
				result = tRs.getString("name");
			else
				throw new EntityNotFoundException(
						"Can't find the teaching!");

			return result;
		} finally {
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Returns the instruction corresponding to the past id as a parameter.
	 * 
	 * @param pId
	 *           The teaching id.
	 * @returnReturns the teaching associated with the past id as a parameter.
	 *
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized Teaching getTeachingById(int pId)
			throws ConnectionException, SQLException, EntityNotFoundException,
			InvalidValueException {
		Teaching result = null;
		Connection connect = null;
		try {

			if (pId <= 0)
				throw new EntityNotFoundException(
						"Unable to find teaching!");

			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			//Prepare SQL string
			String sql = "SELECT * FROM " + ManagerTeaching.TABLE_TEACHING
					+ " WHERE id_teaching = " + Utility.isNull(pId);

			// Send Query to DataBase
			ResultSet pRs = Utility.queryOperation(connect, sql);

			if (pRs.next())
				result = this.loadRecordFromRs(pRs);
			else
				throw new EntityNotFoundException(
						"Unable to find teaching!");

			return result;
		} finally {
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Return the set of all the teachings in the database.
 *
	 * @return A collection of teachings returns.
	 * 
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 * @throws EntityNotFoundException
	 */
	public synchronized Collection<Teaching> getTeachings()
			throws ConnectionException, SQLException, InvalidValueException,
			EntityNotFoundException {
		Collection<Teaching> result = null;
		Connection connect = null;

		try {
			// Prepare SQL string
			String sql = "SELECT * FROM " + ManagerTeaching.TABLE_TEACHING
					+ " ORDER BY name";

			//We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			//Send Query to DataBase
			ResultSet pRs = Utility.queryOperation(connect, sql);

			if (pRs.next())
				result = this.loadRecordsFromRs(pRs);

			return result;
		} finally {
			// releases resources
			DBConnection.releaseConnection(connect);
		}

	}

	/**
	 * Return the set of lessons associated with the corresponding user
	 * to the past id as a parameter.
	 * 
	 * @param pId
	 *            User ID.
	 * @return A collection of teachings returns.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	public synchronized Collection<Teaching> getTeachingsByUserId(int pId)
			throws EntityNotFoundException, ConnectionException, SQLException,
			InvalidValueException {

		Collection<Teaching> result = null;
		Connection connect = null;

		if (pId <= 0)
			throw new EntityNotFoundException("specify user");

		try {
			//Prepare SQL string
			String sql = "SELECT " + ManagerTeaching.TABLE_TEACHING
					+ ".* FROM " + ManagerTeaching.TABLE_TEACHER_CLASSROOM
					+ ", " + ManagerTeaching.TABLE_TEACHING + " WHERE ("
					+ ManagerTeaching.TABLE_TEACHER_CLASSROOM
					+ ".id_teaching = " + ManagerTeaching.TABLE_TEACHING
					+ ".id_teaching AND "
					+ ManagerTeaching.TABLE_TEACHER_CLASSROOM + ".id_user = "
					+ Utility.isNull(pId) + ")" + " ORDER BY name";

			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			//Send Query to DataBase
			ResultSet pRs = Utility.queryOperation(connect, sql);

			if (pRs.next())
				result = this.loadRecordsFromRs(pRs);

			return result;
		} finally {
			// releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Return to the set of teachings that the teacher teaches in the class
	 * 
	 * @param pIdTeacher
	 *            User ID.
	 * @param pIdClass
	 *            class id
	 * @return A collection of teachings returns.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	public synchronized Collection<Teaching> getTeachingsByUserIdClassID(int pIdTeacher,int pIdClass)
			throws EntityNotFoundException, ConnectionException, SQLException,
			InvalidValueException {

		Collection<Teaching> result = null;
		Connection connect = null;

		if (pIdTeacher <= 0)
			throw new EntityNotFoundException("specify user");
		if (pIdClass <= 0)
			throw new EntityNotFoundException("specify the class");

		try {
			//Prepare SQL string
			
			String sql = "SELECT DISTINCT " + ManagerTeaching.TABLE_TEACHING
					+ ".* FROM " + ManagerTeaching.TABLE_TEACHER_CLASSROOM
					+ ", " + ManagerTeaching.TABLE_TEACHING + " WHERE ("
					+ ManagerTeaching.TABLE_TEACHER_CLASSROOM + ".id_user = "
					+ Utility.isNull(pIdTeacher) +" AND "
					+ ManagerTeaching.TABLE_TEACHER_CLASSROOM 
					+ ".id_teaching= " + Utility.isNull(pIdClass)
					+ " AND "
					+ ManagerTeaching.TABLE_TEACHER_CLASSROOM
					+ ".id_teaching = " + ManagerTeaching.TABLE_TEACHING
					+ ".id_teaching "
					+") ORDER BY name";

			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			//Send Query to DataBase
			ResultSet pRs = Utility.queryOperation(connect, sql);

			if (pRs.next())
				result = this.loadRecordsFromRs(pRs);

			return result;
		} finally {
			// releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Check if a teaching has an assigned professor.
	 * 
	 * @param pTeaching
	 *            The teaching to control.
	 * @return Returns true if teaching has a teacher assigned, false
	 *         otherwise.
	 * 
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws InvalidValueException
	 */
	public synchronized boolean hasTeacher(Teaching pTeaching)
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

			//Prepare sql string
			String sql = "SELECT * FROM "
					+ ManagerTeaching.TABLE_TEACHER_CLASSROOM
					+ " WHERE id_teaching = "
					+ Utility.isNull(pTeaching.getId());
			//Send the Query to the database
			ResultSet pRs = Utility.queryOperation(connect, sql);
			if (pRs.next())
				result = true;

			return result;

		} finally {
			// we release resources
			DBConnection.releaseConnection(connect);

		}
	}

	/**
	 * Return the set of teachings associated with the specified class
	 * 
	 * @param pId
	 *            Class ID.
	 * @return A collection of teachings returns.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	public synchronized Collection<Teaching> getTeachingsByClassroomId(int pId)
			throws EntityNotFoundException, ConnectionException, SQLException,
			InvalidValueException {

		Collection<Teaching> result = null;
		Connection connect = null;

		if (pId < 0)
			throw new EntityNotFoundException("specify the Class id!");

		try {
			// Prepare SQL string
			String sql = "SELECT " + ManagerTeaching.TABLE_TEACHING
					+ ".* FROM " + ManagerClassroom.TABLE_CLASSROOM + ", "
					+ ManagerTeaching.TABLE_ADDRESS_TEACHINGS + ", "
					+ ManagerTeaching.TABLE_TEACHING + " WHERE "
					+ ManagerClassroom.TABLE_CLASSROOM + ".id_classroom = "
					+ Utility.isNull(pId) + " AND "
					+ ManagerClassroom.TABLE_CLASSROOM + ".id_address = "
					+ ManagerTeaching.TABLE_ADDRESS_TEACHINGS
					+ ".id_address AND " + ManagerTeaching.TABLE_TEACHING
					+ ".id_teaching= "
					+ ManagerTeaching.TABLE_ADDRESS_TEACHINGS + ".id_teaching ";

			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			// Send Query to DataBase
			ResultSet pRs = Utility.queryOperation(connect, sql);

			if (pRs.next())
				result = this.loadRecordsFromRs(pRs);

			return result;
		} finally {
			// releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Return the set of teachings associated with the specified class
	 * 
	 * @param name
	 *            Class name.
	 * @return A collection of teachings returns.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	public synchronized Collection<Teaching> getTeachingsByClassroomName(
			String name) throws EntityNotFoundException, ConnectionException,
			SQLException, InvalidValueException {

		Collection<Teaching> result = null;
		Connection connect = null;

		if ((name == null) || (name == ""))
			throw new EntityNotFoundException(
					"specify the name of the Class!");

		try {
			// Prepare SQL string
			String sql = "SELECT " + ManagerTeaching.TABLE_TEACHING
					+ ".* FROM " + ManagerClassroom.TABLE_CLASSROOM + ", "
					+ ManagerTeaching.TABLE_ADDRESS_TEACHINGS + ", "
					+ ManagerTeaching.TABLE_TEACHING + " WHERE "
					+ ManagerClassroom.TABLE_CLASSROOM + ".name = "
					+ Utility.isNull(name) + " AND "
					+ ManagerClassroom.TABLE_CLASSROOM + ".id_address = "
					+ ManagerTeaching.TABLE_ADDRESS_TEACHINGS
					+ ".id_address AND " + ManagerTeaching.TABLE_TEACHING
					+ ".id_teaching= "
					+ ManagerTeaching.TABLE_ADDRESS_TEACHINGS + ".id_teaching ";

			// We get a connection to the database
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			// Send Query to DataBase
			ResultSet pRs = Utility.queryOperation(connect, sql);

			if (pRs.next())
				result = this.loadRecordsFromRs(pRs);

			return result;
		} finally {
			// releases resources
			DBConnection.releaseConnection(connect);
		}
	}
	
	public synchronized Collection<Teaching> getTeachingsByIdUserIdClassroom(int pUser, int pClass) throws SQLException,
	EntityNotFoundException, ConnectionException, InvalidValueException {
		
		
		Collection<Teaching> result = null;
		Connection connect = null;
		try {
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			// 	Prepare sql string
			//select teaching.* from teacher_has_classroom AS THC , teaching where thc.id_user = 54 
			//&& thc.id_classroom = 64 && thc.id_teaching = teaching.id_teaching
			
			String sql = "SELECT DISTINCT "
				+ManagerTeaching.TABLE_TEACHING+".*" 
				+" FROM " 
				+ ManagerUser.TABLE_TEACHER_CLASSROOM
				+" , "
				+ManagerTeaching.TABLE_TEACHING
				+ " WHERE "
				+ ManagerUser.TABLE_TEACHER_CLASSROOM
				+ ".id_user = "  
				+ Utility.isNull(pUser)
				+ " AND "
				+ ManagerUser.TABLE_TEACHER_CLASSROOM
				+ ".id_classroom= "
				+ Utility.isNull(pClass)
				+ " AND "
				+ ManagerUser.TABLE_TEACHER_CLASSROOM
				+".id_teaching ="
				+ManagerTeaching.TABLE_TEACHING
				+".id_teaching";
				// Send the Query to the database
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			// Send Query to DataBase
			ResultSet pRs = Utility.queryOperation(connect, sql);

			if (pRs.next())
				result = this.loadRecordsFromRs(pRs);

			return result;
		} finally {
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Lets you read a record from the ResultSet.
	 * 
	 * @param pRs
	 *            The result of the query.
	 * @return Return the teaching read.
	 * 
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	private Teaching loadRecordFromRs(ResultSet pRs) throws SQLException,
			InvalidValueException {
		Teaching teaching = new Teaching();
		teaching.setName(pRs.getString(("name")));
		teaching.setId(pRs.getInt("id_teaching"));

		return teaching;
	}

	/**
	 * Allows you to read records from the ResultSet.
	 * 
	 * @param pRs
	 *            The result of the query.
	 * @return The collection of teachings read returns.
	 * 
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	private Collection<Teaching> loadRecordsFromRs(ResultSet pRs)
			throws SQLException, InvalidValueException {
		Collection<Teaching> result = new Vector<Teaching>();
		do {
			result.add(loadRecordFromRs(pRs));
		} while (pRs.next());
		return result;
	}

}
