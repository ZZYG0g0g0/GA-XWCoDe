package smos.storage;

import smos.bean.Teaching;
import smos.bean.UserListItem;
import smos.bean.Votes;
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
 * Class vote manager.
 *
 * @author Luigi Colangelo
 * @version 1.0
 * 
 *          2009 � Copyright by SMOS
 */

public class ManagerVotes {
	private static ManagerVotes instance;

	/**
	 * The name of the rating table.
	 */
	public static final String TABLE_VOTES = "votes";

	/**
	 * The class builder.
	 */
	public ManagerVotes() {
		super();
	}

	/**
	 * The only instance of the existing vote returns.
	 * 
	 * @return Return the class instance.
	 */
	public static synchronized ManagerVotes getInstance() {
		if (instance == null) {
			instance = new ManagerVotes();
		}
		return instance;
	}

	/**
	 * Check for voting in the database.
	 * 
	 * @param pVotes
	 *            the vote to be checked.
	 * @return Returns true if the passed grade exists as a parameter,
	 *         otherwise false.
	 * 
	 * @throws MandatoryFieldException
	 * @throws SQLException
	 * @throws ConnectionException
	 */
	public synchronized boolean exists(Votes pVotes)
			throws MandatoryFieldException, ConnectionException, SQLException {

		boolean result = false;
		Connection connect = null;

		if (pVotes.getId_votes() == 0)
			throw new MandatoryFieldException("Specify the id.");
		try {
			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			String sql = "SELECT * FROM " + ManagerVotes.TABLE_VOTES
					+ " WHERE id_votes = " + Utility.isNull(pVotes.getId_votes());

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
	 * Inserts a new vote in the Votes table.
	 * 
	 * @param pVotes
	 *            the vote to be inserted.
	 * 
	 * @throws SQLException
	 * @throws ConnectionException
	 * @throws MandatoryFieldException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized void insert(Votes pVotes)
			throws MandatoryFieldException, ConnectionException, SQLException,
			EntityNotFoundException, InvalidValueException {
		Connection connect = null;
		try {
			

			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();
			//Prepare Sql string
			String sql = "INSERT INTO " + ManagerVotes.TABLE_VOTES
			        + " (id_user, id_teaching, written, oral, laboratory, AccademicYear, turn) "
					+ "VALUES ("
					+ Utility.isNull(pVotes.getId_user())
					+", "
					+ Utility.isNull(pVotes.getTeaching()) 
					+", "
					+ Utility.isNull(pVotes.getWritten())
					+", "
					+ Utility.isNull(pVotes.getOral())
					+", "
					+ Utility.isNull(pVotes.getLaboratory())
					+", "
					+ Utility.isNull(pVotes.getAccademicYear())
					+", "
					+ Utility.isNull(pVotes.getTurn())+ " )";

			Utility.executeOperation(connect, sql);

			pVotes.setId_votes(Utility.getMaxValue("id_votes",
					ManagerVotes.TABLE_VOTES));

		} finally {
			// releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Update a vote in the votes table.
	 * 
	 * @param pVotes
	 *            A vote to be changed
	 *
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws MandatoryFieldException
	 */
	public synchronized void update(Votes pVotes)
			throws ConnectionException, SQLException, EntityNotFoundException,
			MandatoryFieldException {
		Connection connect = null;

		try {
			if (pVotes.getId_votes() <= 0)
				throw new EntityNotFoundException(
						"Unable to find the vote!");

			if (pVotes.getId_user() <= 0)
				throw new MandatoryFieldException("Specify the vote user");
			if (pVotes.getTeaching() <= 0)
				throw new MandatoryFieldException("Specify the teaching of voting");
			if (pVotes.getAccademicYear() <= 0)
				throw new MandatoryFieldException("Specify the academic year");
			if (pVotes.getTurn() < 0)
				throw new MandatoryFieldException("Specify semester");
			//Prepare SQL string
			String sql = "UPDATE " + ManagerVotes.TABLE_VOTES + " SET"
					+ " id_user = " + Utility.isNull(pVotes.getId_user())+","+" id_teaching= "
					+ Utility.isNull(pVotes.getTeaching())+","+" written= "
					+ Utility.isNull(pVotes.getWritten())+","+" oral= "
					+ Utility.isNull(pVotes.getOral())+","+" laboratory= "
					+ Utility.isNull(pVotes.getLaboratory())+","+" accademicYear= "
					+ Utility.isNull(pVotes.getAccademicYear())+","+" turn="
					+ Utility.isNull(pVotes.getTurn())
					+ " WHERE id_votes = "
					+ Utility.isNull(pVotes.getId_votes());

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
	 * Check if a student passed as a parameter has a grade assigned in the past teaching
	 * as a parameter in the past year as a parameter and in the past quarter as a parameter
	 *
	 *
	 * @param pTeaching
	 * The teaching to be controlled.
	 * @param pUserListItem
	 * The student to check
	 *
	 * @return Return the id of the vote -1 otherwise
	 * 
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws InvalidValueException
	 */
	public synchronized int getIdVotes(Teaching pTeaching, int academicYear, int turn, UserListItem pUser)
			throws SQLException, EntityNotFoundException, ConnectionException,
			InvalidValueException {
		Connection connect = null;
		int result = -1;
		Votes v = null;
		if (pTeaching.getId() <= 0)
			throw new EntityNotFoundException("Specify teaching");
		if (pUser.getId() <=0 )
			throw new EntityNotFoundException("Specify user");
		try {
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			//Prepare sql string
			String sql = "SELECT * FROM "
					+ ManagerVotes.TABLE_VOTES
					+ " WHERE id_teaching = "
					+ Utility.isNull(pTeaching.getId())
					+ " AND "
					+ ManagerVotes.TABLE_VOTES
					+ ".AccademicYear= "
					+ Utility.isNull(academicYear)
					+ " AND "
					+ ManagerVotes.TABLE_VOTES
					+ ".turn= "
					+ Utility.isNull(turn)
					+ " AND "
					+ ManagerVotes.TABLE_VOTES
					+ ".id_user= "
					+ Utility.isNull(pUser.getId());
			// Send the Query to the database
			ResultSet pRs = Utility.queryOperation(connect, sql);
			if (pRs.next()){
				v = this.loadRecordFromRs(pRs);
				result =v.getId_votes();
				
			}

			return result;

		} finally {
			//we release resources
			DBConnection.releaseConnection(connect);

		}
	}

	/**
	 * Delete a vote from the votes table.
	 *
	 * @param pVotes
	 * The vote to be eliminated.
	 * 
	 * @throws MandatoryFieldException
	 * @throws EntityNotFoundException
	 * @throws SQLException
	 * @throws ConnectionException
	 * @throws InvalidValueException
	 * 
	 */
	public synchronized void delete(Votes pVotes)
			throws ConnectionException, SQLException, EntityNotFoundException,
			MandatoryFieldException, InvalidValueException {
		Connection connect = null;

		try {
			// ManagerTeaching.getInstance().teachingOnDeleteCascade(pTeaching);
			connect = DBConnection.getConnection();
			// Prepare SQL string
			String sql = "DELETE FROM " + ManagerVotes.TABLE_VOTES
					+ " WHERE id_votes = "
					+ Utility.isNull(pVotes.getId_votes());

			Utility.executeOperation(connect, sql);
		} finally {
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Returns the teaching id corresponding to the id of the past grade as
	 * parameter.
	 * 
	 * @param pId
	 * The id of the vote.
	 * @return Returns the teaching id.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	
	/**
	 * Returns the instruction corresponding to the past id as a parameter.
	 *
	 * @param pId
	 * The teaching id.
	 * @return Returns the teaching associated with the past id as a parameter.
	 * 
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws EntityNotFoundException
	 * @throws InvalidValueException
	 */
	public synchronized Votes getVotesById(int pId)
			throws ConnectionException, SQLException, EntityNotFoundException,
			InvalidValueException {
		Votes result = null;
		Connection connect = null;
		try {

			if (pId <= 0)
				throw new EntityNotFoundException(
						"Unable to find the vote!");

			//We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			// Prepare SQL string
			String sql = "SELECT * FROM " + ManagerVotes.TABLE_VOTES
					+ " WHERE id_votes = " + Utility.isNull(pId);

			// Send Query to DataBase
			ResultSet pRs = Utility.queryOperation(connect, sql);

			if (pRs.next())
				result = this.loadRecordFromRs(pRs);
			else
				throw new EntityNotFoundException(
						"Can't find the teaching!");

			return result;
		} finally {
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}
	
	public synchronized String getTeachingIdByVotesId(int pId)
			throws EntityNotFoundException, ConnectionException, SQLException {
		String result;
		Connection connect = null;
		try {
			// If the id has not been provided we return an error code
			if (pId <= 0)
				throw new EntityNotFoundException(
						"Unable to find the vote!");

			/*
			 * Preparing SQL string to retrieve information
 			 * corresponding to the id of the past teaching as a parameter
			 */
			String tSql = "SELECT id_teaching FROM " + ManagerVotes.TABLE_VOTES
			        
					+ " WHERE id_votes = " + Utility.isNull(pId);

			// We get a connection to the database.
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			// Send Query to DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);

			if (tRs.next())
				result = tRs.getString("id_teaching");
			else
				throw new EntityNotFoundException(
						"Unable to find the vote!");

			return result;
		} finally {
			DBConnection.releaseConnection(connect);
		}
	}
	

	/**
	 * Returns the set of all votes in the database.
	 *
	 * @return Returns a collection of votes.
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 * @throws EntityNotFoundException
	 */
	public synchronized Collection<Votes> getVotes()
			throws ConnectionException, SQLException, InvalidValueException,
			EntityNotFoundException {
		Collection<Votes> result = null;
		Connection connect = null;

		try {
			// Prepare SQL string
			String sql = "SELECT * FROM " + ManagerVotes.TABLE_VOTES
					+ " ORDER BY id_votes";

			//We get a connection to the database.
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
	 * Returns the set of votes associated with the corresponding user
	 * to the id passed as a parameter.
	 *
	 * @param pId
	 * The user id.
	 * @return Returns a collection of votes.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	public synchronized Collection<Votes> getVotesByUserId(int pId)
			throws EntityNotFoundException, ConnectionException, SQLException,
			InvalidValueException {

		Collection<Votes> result = null;
		Connection connect = null;

		if (pId <= 0)
			throw new EntityNotFoundException("specify user");

		try {
			// Prepare SQL string
			String sql = "SELECT " + ManagerVotes.TABLE_VOTES
					+ ".* FROM " + ManagerVotes.TABLE_VOTES
				    + " WHERE ("
					+ ManagerVotes.TABLE_VOTES + ".id_user = "
					+ Utility.isNull(pId) + ")" + " ORDER BY id_user";

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
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Returns the set of votes associated with the corresponding user
	 * to the id passed as a parameter.
	 *
	 * @param pId
	 * The user id.
	 * @return Returns a collection of votes.
		* 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	public synchronized Collection<Votes> getVotesByUserIdYearTurn(int pId,int pYear, int pTurn)
			throws EntityNotFoundException, ConnectionException, SQLException,
			InvalidValueException {

		Collection<Votes> result = null;
		Connection connect = null;

		if (pId <= 0)
			throw new EntityNotFoundException("specify user");

		try {
			// Prepare SQL string
			String sql = "SELECT " + ManagerVotes.TABLE_VOTES
					+ ".* FROM " + ManagerVotes.TABLE_VOTES
				    + " WHERE ("
					+ ManagerVotes.TABLE_VOTES + ".id_user = "
					+ Utility.isNull(pId)+" AND "
					+ ManagerVotes.TABLE_VOTES + ".accademicYear = "
					+ Utility.isNull(pYear)+" AND " 
					+ManagerVotes.TABLE_VOTES + ".turn = "
					+ Utility.isNull(pTurn)+ ")" + " ORDER BY id_user";

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
			//releases resources
			DBConnection.releaseConnection(connect);
		}
	}
	
	public synchronized void deleteVotesByUserIdYearTurn(int pId,int pYear, int pTurn)
	throws EntityNotFoundException, ConnectionException, SQLException,
	InvalidValueException {

Connection connect = null;

if (pId <= 0)
	throw new EntityNotFoundException("specificare l'utente");

try {
	// Prepare SQL string
	String sql = "DELETE "+ ManagerVotes.TABLE_VOTES+" FROM " + ManagerVotes.TABLE_VOTES
		    + " WHERE ("
			+ ManagerVotes.TABLE_VOTES + ".id_user="
			+ Utility.isNull(pId)+" AND "
			+ ManagerVotes.TABLE_VOTES + ".AccademicYear="
			+ Utility.isNull(pYear)+" AND " 
			+ManagerVotes.TABLE_VOTES + ".turn="
			+ Utility.isNull(pTurn)+ ")";

	// We get a connection to the database.
	connect = DBConnection.getConnection();
	if (connect == null)
		throw new ConnectionException();

	// Send Query to DataBase
	 Utility.executeOperation(connect, sql);
     
} finally {
	// releases resources
	DBConnection.releaseConnection(connect);
}
}
	
	
	/**
	 * Lets you read a record from the ResultSet.
	 *
	 * @param pRs
	 * The result of the query.
	 * @return Return the read vote.
	 * 
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	private Votes loadRecordFromRs(ResultSet pRs) throws SQLException,
			InvalidValueException {
		Votes votes = new Votes();
		votes.setId_votes(pRs.getInt(("id_votes")));
		votes.setId_user(pRs.getInt("id_user"));
		votes.setTeaching(pRs.getInt("id_teaching"));
		votes.setWritten(pRs.getInt("written"));
		votes.setOral(pRs.getInt("oral"));
		votes.setLaboratory(pRs.getInt("laboratory"));
		votes.setAccademicYear(pRs.getInt("AccademicYear"));
		votes.setTurn(pRs.getInt("turn"));

		return votes;
	}

	/**
	 * Allows you to read records from the ResultSet.
	 *
	 * @param pRs
	 * The result of the query.
	 * @return Returns the collection of read teachings.
	 * 
	 * @throws SQLException
	 * @throws InvalidValueException
	 */
	private Collection<Votes> loadRecordsFromRs(ResultSet pRs)
			throws SQLException, InvalidValueException {
		Collection<Votes> result = new Vector<Votes>();
		do {
			result.add(loadRecordFromRs(pRs));
		} while (pRs.next());
		return result;
	}

}
