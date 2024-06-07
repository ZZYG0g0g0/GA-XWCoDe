package smos.bean;

import java.io.Serializable;
import java.util.Date;

public class Justify implements Serializable {

	/**
	 * Class modeling a justifies for an absence
	 * @author Nicola Pisanti
	 * @version 1.0
	 * 
	 */
	private static final long serialVersionUID = -4726381877687167661L;

	private int idJustify;
	private int idUser;
	private Date dateJustify;
	private int academicYear;
	
	/**
	 * Method that returns the id of the justify
	 * @return an integer representing the id of the justify
	 */
	public int getIdJustify() {
		return idJustify;
	}
	/**
	 * Method setting the id of the justify
	 * @param an integer representing the id to be set
	 */
	public void setIdJustify(int pIdJustify) {
		this.idJustify = pIdJustify;
	}
	/**
	 * Method returns the student id relative to justify
	 * @return an integer representing the student id
	 */
	public int getIdUser() {
		return idUser;
	}
	/**
	 * Method setting student id relative to justify
     * @param an integer representing the id to be set
	 */
	public void setIdUser(int pIdUser) {
		this.idUser = pIdUser;
	}
	/**
	 * Method returning the date on which absence was justified
     * @return a string representing the justified date
	 */
	public Date getDateJustify() {
		return dateJustify;
	}
	/**
	 * Method setting the date on which absence was justified
     * @param a string representing the date to set
	 */
	public void setDateJustify(Date pDateJustify) {
		this.dateJustify = pDateJustify;
	}
	/**
	 * Method that returns the academic year relative to the justifies
     * @return an integer representing the year in which the academic year began
	 */
	public int getAcademicYear() {
		return academicYear;
	}
	/**
	 * Method setting the academic year relating to the justification
     * @param an integer representing the academic year to be set
	 */
	public void setAcademicYear(int pAcademicYear) {
		this.academicYear = pAcademicYear;
	}
	
	
	
}
